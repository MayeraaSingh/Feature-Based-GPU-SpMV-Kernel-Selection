"""Tests for selector/model.py, baselines.py and train.py (Milestone 3).

Uses SYNTHETIC data only. In the fixture, kernel_A wins on low-variance
matrices and kernel_C on high-variance ones, so a forest can separate them.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import score_selections, summarize_scores
from selector.baselines import (
    best_fixed_on_train,
    fixed_selections,
    majority_kernel_on_train,
    train_kernel_table,
)
from selector.data import FEATURE_COLUMNS, Dataset, build_dataset
from selector.labels import oracle_table
from selector.model import KernelSelector
from selector.train import (
    TrainConfig,
    expand_grid,
    train_selector,
    write_training_outputs,
)
from tests.fixtures.synthetic_timings import make_tables, make_timings, write_files

SMALL_GRID = {"n_estimators": [10, 20], "max_depth": [None, 2]}
CFG = TrainConfig(seed=3, grid=SMALL_GRID)


def make_dataset(tmp_path, timings=None, tables=None):
    t = make_tables()
    t.update(tables or {})
    t["timings"] = timings if timings is not None else make_timings(t["features"])
    p = write_files(tmp_path, t)
    return build_dataset(p["features"], p["splits"], p["catalog"], p["timings"]), p


def xy(ds, split="train"):
    f = ds.features_for(split)
    oracle, _ = oracle_table(ds.timings_for(split))
    y = oracle.set_index("matrix_id").loc[f["matrix_id"], "oracle_kernel"].tolist()
    return f, y


# ------------------------------------------------------------------ #
# KernelSelector                                                       #
# ------------------------------------------------------------------ #

class TestKernelSelector:
    def test_learns_separable_synthetic_data(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        sel = KernelSelector(seed=0).fit(X, y)
        assert sel.predict(X).tolist() == y == ["kernel_A", "kernel_A", "kernel_C", "kernel_C"]
        Xt, yt = xy(ds, "test")
        assert sel.predict(Xt).tolist() == yt

    def test_deterministic_with_same_seed(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        a = KernelSelector(seed=5, n_estimators=30).fit(X, y)
        b = KernelSelector(seed=5, n_estimators=30).fit(X, y)
        Xt = ds.features_for("test")
        pd.testing.assert_frame_equal(a.predict_proba(Xt), b.predict_proba(Xt))

    def test_save_load_round_trip(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        a = KernelSelector(seed=1, n_estimators=25, max_depth=3, class_weight="balanced").fit(X, y)
        path = a.save(tmp_path / "sub" / "m.joblib")
        b = KernelSelector.load(path)
        Xt = ds.features_for("test")
        assert a.predict(Xt).tolist() == b.predict(Xt).tolist()
        pd.testing.assert_frame_equal(a.predict_proba(Xt), b.predict_proba(Xt))
        assert (b.feature_columns, b.seed, b.params, b.versions) == (
            a.feature_columns, a.seed, a.params, a.versions)

    def test_load_warns_on_version_mismatch(self, tmp_path):
        import joblib
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        path = KernelSelector().fit(X, y).save(tmp_path / "m.joblib")
        blob = joblib.load(path)
        blob["versions"]["scikit-learn"] = "0.0.1"
        joblib.dump(blob, path)
        with pytest.warns(RuntimeWarning, match="different library versions"):
            KernelSelector.load(path)

    def test_load_no_warning_when_versions_match(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        path = KernelSelector().fit(X, y).save(tmp_path / "m.joblib")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            KernelSelector.load(path)

    # -- feature order / validation ---------------------------------- #

    def test_column_order_in_input_does_not_matter(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        sel = KernelSelector().fit(X, y)
        Xt = ds.features_for("test")
        shuffled = Xt[list(reversed(Xt.columns))]
        assert sel.predict(Xt).tolist() == sel.predict(shuffled).tolist()

    def test_model_uses_its_own_feature_order(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        sel = KernelSelector(feature_columns=["variance_row_nnz", "mean_row_nnz"]).fit(X, y)
        assert sel.feature_columns == ["variance_row_nnz", "mean_row_nnz"]
        sel.predict(X)  # extra columns in X are ignored

    def test_missing_feature_column_raises(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        sel = KernelSelector().fit(X, y)
        with pytest.raises(ValueError, match="missing feature column.*nnz"):
            sel.predict(X.drop(columns=["nnz"]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_input_raises(self, tmp_path, bad):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        sel = KernelSelector().fit(X, y)
        X = X.copy()
        X["mean_row_nnz"] = X["mean_row_nnz"].astype(float)
        X.loc[0, "mean_row_nnz"] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            sel.predict(X)

    def test_non_dataframe_rejected(self):
        with pytest.raises(TypeError):
            KernelSelector().fit(np.zeros((2, 7)), ["a", "b"])

    # -- classes / edge cases ---------------------------------------- #

    def test_classes_are_only_training_oracles(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        sel = KernelSelector().fit(X, y)
        assert sel.classes_ == ["kernel_A", "kernel_C"]      # kernel_B never won
        assert set(sel.predict(ds.features)) <= {"kernel_A", "kernel_C"}
        proba = sel.predict_proba(ds.features_for("test"))
        assert list(proba.columns) == sel.classes_
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_single_class_training_works(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, _ = xy(ds)
        sel = KernelSelector().fit(X, ["kernel_A"] * len(X))
        assert sel.classes_ == ["kernel_A"]
        assert set(sel.predict(ds.features_for("test"))) == {"kernel_A"}
        assert sel.predict_proba(X).shape == (len(X), 1)

    def test_predict_on_empty_frame(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        sel = KernelSelector().fit(X, y)
        assert len(sel.predict(X.iloc[0:0])) == 0
        assert sel.predict_proba(X.iloc[0:0]).shape == (0, 2)

    def test_feature_importances_named(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        imp = KernelSelector().fit(X, y).feature_importances()
        assert list(imp.index) == FEATURE_COLUMNS and imp.sum() == pytest.approx(1.0)

    def test_misuse_errors(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        X, y = xy(ds)
        with pytest.raises(RuntimeError, match="not fitted"):
            KernelSelector().predict(X)
        with pytest.raises(ValueError, match="unknown model parameter"):
            KernelSelector(bogus=1)
        with pytest.raises(ValueError, match="rows but y has"):
            KernelSelector().fit(X, y[:-1])
        with pytest.raises(ValueError, match="zero training"):
            KernelSelector().fit(X.iloc[0:0], [])
        with pytest.raises(ValueError, match="non-empty kernel-name strings"):
            KernelSelector().fit(X, [1, 2, 3, 4])


# ------------------------------------------------------------------ #
# Baselines                                                            #
# ------------------------------------------------------------------ #

class TestBaselines:
    def test_best_fixed_is_lowest_total_train_time(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        kernel, table = best_fixed_on_train(ds)
        totals = ds.timings_for("train").groupby("kernel_name")["kernel_time_ms"].sum()
        assert kernel == totals.idxmin()
        assert table.set_index("kernel_name")["total_time_ms"].to_dict() == pytest.approx(totals.to_dict())

    def test_kernel_failing_correctness_is_not_eligible(self, tmp_path):
        t = make_timings(make_tables()["features"])
        # Make kernel_C the cheapest overall but wrong on one train matrix.
        t.loc[t["kernel_name"] == "kernel_C", "kernel_time_ms"] = 1e-6
        t.loc[(t["kernel_name"] == "kernel_C") & (t["matrix_id"] == "m01"), "correctness_pass"] = False
        ds, _ = make_dataset(tmp_path, t)
        kernel, table = best_fixed_on_train(ds)
        assert kernel != "kernel_C"
        row = table.set_index("kernel_name").loc["kernel_C"]
        assert (row.n_incorrect, row.eligible) == (1, False)

    def test_no_eligible_kernel_raises(self, tmp_path):
        t = make_timings(make_tables()["features"])
        for k, m in (("kernel_A", "m01"), ("kernel_B", "m02"), ("kernel_C", "m03")):
            t.loc[(t["kernel_name"] == k) & (t["matrix_id"] == m), "correctness_pass"] = False
        ds, _ = make_dataset(tmp_path, t)
        with pytest.raises(ValueError, match="no kernel is correct on every train matrix"):
            best_fixed_on_train(ds)

    def test_baseline_decided_on_train_only(self, tmp_path):
        base = make_timings(make_tables()["features"])
        tampered = base.copy()
        held = tampered["matrix_id"].isin(["m05", "m06", "m07", "m08"])
        tampered.loc[held & (tampered["kernel_name"] == "kernel_B"), "kernel_time_ms"] = 1e-9
        a = best_fixed_on_train(make_dataset(tmp_path / "a", base)[0])[0]
        b = best_fixed_on_train(make_dataset(tmp_path / "b", tampered)[0])[0]
        assert a == b

    def test_majority_kernel(self, tmp_path):
        ds, _ = make_dataset(tmp_path)  # train labels: A, A, C, C -> tie -> alphabetical
        assert majority_kernel_on_train(ds) == "kernel_A"
        t = make_timings(make_tables()["features"])
        t.loc[(t.matrix_id == "m01") & (t.kernel_name == "kernel_C"), "kernel_time_ms"] = 1e-6
        assert majority_kernel_on_train(make_dataset(tmp_path / "x", t)[0]) == "kernel_C"

    def test_fixed_selections_score_like_any_selector(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        ids = ds.matrix_ids("test")
        timings = ds.timings_for("test")
        oracle, _ = oracle_table(timings)
        scores = score_selections(fixed_selections(ids, "kernel_A"), timings, oracle)
        assert scores["selected_kernel"].tolist() == ["kernel_A", "kernel_A"]
        summ = summarize_scores(scores)
        assert summ["accuracy"] == 0.5          # right on m07 only
        assert summ["mean_loss"] > 0

    def test_train_kernel_table_columns(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        table = train_kernel_table(ds)
        assert list(table.columns) == [
            "kernel_name", "total_time_ms", "n_matrices", "n_incorrect", "eligible"]
        assert table["n_matrices"].tolist() == [4, 4, 4]


# ------------------------------------------------------------------ #
# Training                                                             #
# ------------------------------------------------------------------ #

class GuardedDataset(Dataset):
    """A dataset that raises if anything asks for the test split."""

    @staticmethod
    def _check_split(split):
        if split == "test":
            raise AssertionError("training touched the TEST split")
        Dataset._check_split(split)


def guarded(ds):
    return GuardedDataset(ds.features, ds.candidate_kernels, ds.excluded, ds._timings)


class TestTrainSelector:
    def test_end_to_end_on_synthetic(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        res = train_selector(ds, CFG)
        assert res.selector.classes_ == ["kernel_A", "kernel_C"]
        assert len(res.tuning) == 4 and res.tuning["chosen"].sum() == 1
        assert res.card["n_train_matrices_used"] == 4
        assert res.card["synthetic_data"] is False  # no input_paths given
        assert res.train_no_oracle.empty

    def test_chosen_row_is_best_under_documented_ranking(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        t = train_selector(ds, CFG).tuning
        chosen = t[t["chosen"]].iloc[0]
        for _, r in t.iterrows():
            assert (r.n_incorrect_selection, r.mean_val_loss) >= (
                chosen.n_incorrect_selection, chosen.mean_val_loss)

    def test_deterministic_across_runs(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        a, b = train_selector(ds, CFG), train_selector(ds, CFG)
        assert a.card == b.card and a.best_params == b.best_params
        pd.testing.assert_frame_equal(a.tuning, b.tuning)
        Xt = ds.features_for("test")
        pd.testing.assert_frame_equal(a.selector.predict_proba(Xt), b.selector.predict_proba(Xt))

    def test_never_touches_test_split(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        train_selector(guarded(ds), CFG)                          # would raise if it did
        train_selector(guarded(ds), TrainConfig(seed=3, grid=SMALL_GRID, refit_on_train_val=True))
        with pytest.raises(AssertionError):                       # guard itself works
            guarded(ds).timings_for("test")

    def test_tampering_with_test_data_cannot_change_the_model(self, tmp_path):
        base = make_tables()
        timings = make_timings(base["features"])
        ds1, _ = make_dataset(tmp_path / "a", timings)

        bad_t = timings.copy()
        is_test = bad_t["matrix_id"].isin(["m07", "m08"])
        flipped = bad_t.loc[is_test, "kernel_time_ms"].iloc[::-1].to_numpy()
        bad_t.loc[is_test, "kernel_time_ms"] = flipped
        bad_t.loc[is_test, "correctness_pass"] = False
        bad_f = base["features"].copy()
        bad_f.loc[bad_f.matrix_id.isin(["m07", "m08"]), "variance_row_nnz"] = 12345.0
        ds2, _ = make_dataset(tmp_path / "b", bad_t, {"features": bad_f})

        a, b = train_selector(ds1, CFG), train_selector(ds2, CFG)
        pd.testing.assert_frame_equal(a.tuning, b.tuning)
        assert a.best_params == b.best_params
        probe = ds1.features_for("train")
        pd.testing.assert_frame_equal(a.selector.predict_proba(probe), b.selector.predict_proba(probe))

    def test_refit_on_train_val_uses_more_matrices(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        res = train_selector(ds, TrainConfig(seed=3, grid=SMALL_GRID, refit_on_train_val=True))
        assert res.card["n_final_fit_matrices"] == 6 and res.card["n_train_matrices_used"] == 4

    def test_no_validation_data_cannot_tune_a_grid(self, tmp_path):
        t = make_tables()
        t["splits"] = t["splits"].replace({"split": {"val": "train"}})
        ds, _ = make_dataset(tmp_path, tables=t)
        with pytest.raises(ValueError, match="no validation matrices"):
            train_selector(ds, CFG)
        res = train_selector(ds, TrainConfig(seed=3, grid={"n_estimators": [10]}))   # single setting is fine
        assert np.isnan(res.tuning["mean_val_loss"].iloc[0])

    def test_train_without_any_oracle_raises(self, tmp_path):
        t = make_timings(make_tables()["features"])
        t.loc[t["matrix_id"].isin(["m01", "m02", "m03", "m04"]), "correctness_pass"] = False
        ds, _ = make_dataset(tmp_path, t)
        with pytest.raises(ValueError, match="no train matrices with an oracle"):
            train_selector(ds, CFG)

    def test_train_matrix_without_oracle_is_reported_not_used(self, tmp_path):
        t = make_timings(make_tables()["features"])
        t.loc[t["matrix_id"] == "m02", "correctness_pass"] = False
        ds, _ = make_dataset(tmp_path, t)
        res = train_selector(ds, CFG)
        assert res.card["n_train_matrices_used"] == 3 and res.card["n_train_without_oracle"] == 1
        assert res.train_no_oracle["matrix_id"].tolist() == ["m02"]

    def test_selector_never_picks_unlabelled_kernel(self, tmp_path):
        ds, _ = make_dataset(tmp_path)
        res = train_selector(ds, CFG)
        assert set(res.selector.predict(ds.features)) <= set(ds.candidate_kernels)


class TestGridAndOutputs:
    def test_expand_grid_is_deterministic_and_complete(self):
        g = expand_grid({"max_depth": [None, 2], "n_estimators": [5, 6, 7]})
        assert len(g) == 6 and g[0] == {"max_depth": None, "n_estimators": 5}
        assert g == expand_grid({"n_estimators": [5, 6, 7], "max_depth": [None, 2]})

    def test_expand_grid_errors(self):
        with pytest.raises(ValueError, match="unknown grid parameter"):
            expand_grid({"bogus": [1]})
        with pytest.raises(ValueError, match="at least one value"):
            expand_grid({"n_estimators": []})
        assert expand_grid({}) == [{}]

    def test_outputs_written_and_synthetic_stamped(self, tmp_path):
        ds, paths = make_dataset(tmp_path / "data")
        res = train_selector(ds, CFG, input_paths=paths)
        assert res.card["synthetic_data"] is True
        hashes = res.card["input_sha256"]
        assert set(hashes) == set(paths) and all(len(h) == 64 for h in hashes.values())
        out = write_training_outputs(res, tmp_path / "run")
        assert all(p.is_file() for p in out.values())
        card = json.loads(out["model_card"].read_text())
        assert card["classes"] == ["kernel_A", "kernel_C"] and card["versions"]["scikit-learn"]
        loaded = KernelSelector.load(out["model"])
        Xt = ds.features_for("test")
        assert loaded.predict(Xt).tolist() == res.selector.predict(Xt).tolist()
        assert len(pd.read_csv(out["tuning"])) == 4

    def test_input_hash_changes_when_file_changes(self, tmp_path):
        ds, paths = make_dataset(tmp_path / "data")
        h1 = train_selector(ds, CFG, input_paths=paths).card["input_sha256"]["timings"]
        with open(paths["timings"], "a") as fh:
            fh.write("\n")
        h2 = train_selector(ds, CFG, input_paths=paths).card["input_sha256"]["timings"]
        assert h1 != h2
