"""Milestone 4 tests: evaluate -> tables -> plots -> config -> CLI scripts.

SYNTHETIC data only (tests.fixtures.synthetic_timings). Nothing here is a GPU result.
"""

import contextlib
import functools
import importlib.util
import io
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation.evaluate import (
    ORACLE, PER_MATRIX_COLUMNS, SELECTOR, evaluate, fixed_method, method_order,
)
from evaluation.plots import make_all_plots
from evaluation.tables import (
    SYNTHETIC_BANNER, confusion_table, group_table, overall_table, to_markdown, write_report_tables,
)
from selector.config import (
    ConfigError, DEFAULT_CONFIG_PATH, apply_overrides, is_synthetic, latest_run_dir, load_config,
    make_run_id, train_config,
)
from selector.data import build_dataset
from selector.model import KernelSelector
from selector.train import TrainConfig, train_selector
from tests.fixtures.synthetic_timings import make_tables, make_timings, write_files

ROOT = Path(__file__).resolve().parents[1]
SMALL = TrainConfig(seed=1, grid={"n_estimators": [20], "max_depth": [None]})


def dataset_from(tmp_path, timings=None, tables=None):
    t = make_tables()
    t.update(tables or {})
    t["timings"] = timings if timings is not None else make_timings(t["features"])
    p = write_files(tmp_path, t)
    return build_dataset(p["features"], p["splits"], p["catalog"], p["timings"]), p


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ds, paths = dataset_from(tmp_path_factory.mktemp("data"))
    res = train_selector(ds, SMALL, input_paths=paths)
    return ds, res, evaluate(res.selector, ds, "test")


# ------------------------------------------------------------------ #
# evaluate()                                                           #
# ------------------------------------------------------------------ #

class TestEvaluate:
    def test_columns_and_methods(self, trained):
        ds, res, ev = trained
        assert list(ev.per_matrix.columns) == PER_MATRIX_COLUMNS
        assert set(ev.per_matrix["method"]) == {
            SELECTOR, ORACLE, "fixed_kernel_A", "fixed_kernel_B", "fixed_kernel_C"}
        assert set(ev.per_matrix["matrix_id"]) == {"m07", "m08"}
        assert set(ev.per_matrix["split"]) == {"test"}
        assert ev.per_matrix.groupby("method").size().eq(2).all()

    def test_oracle_rows_are_perfect(self, trained):
        pm = trained[2].per_matrix
        o = pm[pm["method"] == ORACLE]
        assert (o["oracle_relative_loss"] == 0).all() and o["selection_correct"].all()

    def test_selector_matches_oracle_on_separable_synthetic_data(self, trained):
        pm = trained[2].per_matrix
        s = pm[pm["method"] == SELECTOR]
        assert s["selection_correct"].all()
        assert s["selected_kernel"].tolist() == s["oracle_kernel"].tolist()

    def test_best_fixed_comes_from_train_and_is_ordered_second(self, trained):
        ds, _, ev = trained
        train_totals = ds.timings_for("train").groupby("kernel_name")["kernel_time_ms"].sum()
        assert ev.best_fixed_kernel == train_totals.idxmin()
        order = method_order(ev.per_matrix, ev.best_fixed_kernel)
        assert order[0] == SELECTOR and order[1] == fixed_method(ev.best_fixed_kernel) and order[-1] == ORACLE

    def test_fixed_rows_hand_check(self, trained):
        ds, _, ev = trained
        t = ds.timings_for("test").set_index(["matrix_id", "kernel_name"])["kernel_time_ms"]
        pm = ev.per_matrix
        row = pm[(pm.method == "fixed_kernel_A") & (pm.matrix_id == "m08")].iloc[0]
        oracle_t = pm[(pm.method == ORACLE) & (pm.matrix_id == "m08")].iloc[0].selected_kernel_time_ms
        assert row.selected_kernel_time_ms == t[("m08", "kernel_A")]
        assert row.oracle_relative_loss == pytest.approx(t[("m08", "kernel_A")] / oracle_t - 1)

    def test_overhead_measured_only_for_selector(self, trained):
        pm = trained[2].per_matrix
        assert (pm.loc[pm.method == SELECTOR, "selection_overhead_ms"] > 0).all()
        assert pm.loc[pm.method != SELECTOR, "selection_overhead_ms"].isna().all()

    def test_matrix_group_carried_through(self, trained):
        pm = trained[2].per_matrix
        assert dict(zip(pm.matrix_id, pm.matrix_group)) == {"m07": "g4", "m08": "g5"}

    def test_test_matrix_without_correct_kernel_is_excluded_and_reported(self, tmp_path):
        t = make_timings(make_tables()["features"])
        t.loc[t.matrix_id == "m08", "correctness_pass"] = False
        ds, _ = dataset_from(tmp_path, t)
        ev = evaluate(train_selector(ds, SMALL).selector, ds, "test")
        assert set(ev.per_matrix["matrix_id"]) == {"m07"}
        assert ev.excluded.to_dict("records") == [
            {"matrix_id": "m08", "split": "test", "reason": "no_correct_kernel",
             "detail": "kernel_A;kernel_B;kernel_C"}]

    def test_candidate_coverage_exclusion_is_reported_for_the_split(self, tmp_path):
        t = make_timings(make_tables()["features"])
        t = t[~((t.matrix_id == "m07") & (t.kernel_name == "kernel_B"))]
        ds, _ = dataset_from(tmp_path, t)
        ev = evaluate(train_selector(ds, SMALL).selector, ds, "test")
        assert ev.excluded["reason"].tolist() == ["missing_candidate_timings"]
        assert set(ev.per_matrix["matrix_id"]) == {"m08"}

    def test_selecting_an_incorrect_kernel_gives_nan_loss(self, tmp_path):
        t = make_timings(make_tables()["features"])
        # Break the kernel the selector will choose for m07 (kernel_A) -> wrong result there.
        t.loc[(t.matrix_id == "m07") & (t.kernel_name == "kernel_A"), "correctness_pass"] = False
        ds, _ = dataset_from(tmp_path, t)
        ev = evaluate(train_selector(ds, SMALL).selector, ds, "test")
        s = ev.per_matrix[(ev.per_matrix.method == SELECTOR) & (ev.per_matrix.matrix_id == "m07")].iloc[0]
        assert not s.selected_correct_flag and not s.selection_correct and math.isnan(s.oracle_relative_loss)

    def test_model_with_unknown_kernel_is_rejected(self, tmp_path):
        ds, _ = dataset_from(tmp_path)
        X = ds.features_for("train")
        bad = KernelSelector().fit(X, ["not_a_kernel"] * len(X))
        with pytest.raises(ValueError, match="not in this dataset"):
            evaluate(bad, ds, "test")

    def test_empty_split_does_not_crash(self, tmp_path):
        t = make_tables()
        t["splits"] = t["splits"].replace({"split": {"test": "val"}})
        ds, _ = dataset_from(tmp_path, tables=t)
        ev = evaluate(train_selector(ds, SMALL).selector, ds, "test")
        assert ev.per_matrix.empty and list(ev.per_matrix.columns) == PER_MATRIX_COLUMNS


# ------------------------------------------------------------------ #
# tables                                                               #
# ------------------------------------------------------------------ #

class TestTables:
    def test_overall_table(self, trained):
        _, _, ev = trained
        o = overall_table(ev.per_matrix, ev.best_fixed_kernel, n_boot=50).set_index("method")
        assert o.loc[SELECTOR, "accuracy"] == 1.0 and o.loc[ORACLE, "mean_loss"] == 0.0
        assert o["is_best_fixed"].sum() == 1 and o.loc[fixed_method(ev.best_fixed_kernel), "is_best_fixed"]
        assert {"within_0pct", "within_25pct", "total_end_to_end_ratio", "n_incorrect_selection"} <= set(o.columns)
        assert o.loc[SELECTOR, "mean_selection_overhead_ms"] > 0

    def test_group_table_selector_vs_best_fixed(self, trained):
        _, _, ev = trained
        g = group_table(ev.per_matrix, ev.best_fixed_kernel, n_boot=50)
        assert set(g["matrix_group"]) == {"g4", "g5"} and set(g["method"]) == {
            SELECTOR, fixed_method(ev.best_fixed_kernel)}
        assert (g["n_matrices"] == 1).all()

    def test_confusion_counts_and_all_candidates_present(self, trained):
        _, _, ev = trained
        c = confusion_table(ev.per_matrix, ev.candidate_kernels)
        assert list(c.index) == list(c.columns) == ["kernel_A", "kernel_B", "kernel_C"]
        assert c.to_numpy().sum() == 2 and c.loc["kernel_B"].sum() == 0
        assert c.loc["kernel_A", "kernel_A"] == 1 and c.loc["kernel_C", "kernel_C"] == 1

    def test_markdown_helper(self):
        md = to_markdown(pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]}), banner="B", note="N")
        assert md.startswith("> **B**") and "| a | b |" in md and "| 1 | 0.5000 |" in md and "| 2 |  |" in md

    @pytest.mark.parametrize("synthetic", [True, False])
    def test_write_report_tables_and_stamping(self, trained, tmp_path, synthetic):
        _, _, ev = trained
        paths = write_report_tables(tmp_path, ev.per_matrix, ev.excluded, ev.best_fixed_kernel,
                                    ev.candidate_kernels, synthetic, n_boot=50)
        for key in ("per_matrix_test", "summary_overall", "summary_by_group", "excluded_matrices",
                    "confusion_matrix"):
            assert paths[key].is_file()
        stamped = pd.read_csv(paths["per_matrix_test"])["provenance"].unique().tolist()
        assert stamped == ([SYNTHETIC_BANNER] if synthetic else ["measured"])
        assert paths["summary_overall_md"].read_text().startswith("> **SYNTHETIC") == synthetic
        assert (tmp_path / "RESULTS_ARE_SYNTHETIC.txt").exists() == synthetic


# ------------------------------------------------------------------ #
# plots                                                                #
# ------------------------------------------------------------------ #

PNG = b"\x89PNG\r\n\x1a\n"


class TestPlots:
    def test_all_six_figures_are_valid_pngs(self, trained, tmp_path):
        _, res, ev = trained
        confusion = confusion_table(ev.per_matrix, ev.candidate_kernels)
        figs = make_all_plots(tmp_path, ev.per_matrix, confusion, res.selector.feature_importances(),
                              ev.best_fixed_kernel, synthetic=True, n_boot=50)
        assert len(figs) == 6
        for p in figs.values():
            assert p.read_bytes()[:8] == PNG and p.stat().st_size > 5000

    def test_plots_survive_nan_losses_and_empty_input(self, trained, tmp_path):
        _, res, ev = trained
        pm = ev.per_matrix.copy()
        bad = (pm.method == SELECTOR) & (pm.matrix_id == "m07")
        pm.loc[bad, ["oracle_relative_loss", "selected_correct_flag", "selection_correct"]] = [np.nan, False, False]
        confusion = confusion_table(pm, ev.candidate_kernels)
        make_all_plots(tmp_path / "nan", pm, confusion, res.selector.feature_importances(),
                       ev.best_fixed_kernel, n_boot=50)
        empty = pm.iloc[0:0]
        figs = make_all_plots(tmp_path / "empty", empty, confusion_table(empty, ev.candidate_kernels),
                              res.selector.feature_importances(), ev.best_fixed_kernel, n_boot=50)
        assert all(p.is_file() for p in figs.values())

    def test_plots_without_a_best_fixed_kernel(self, trained, tmp_path):
        _, res, ev = trained
        make_all_plots(tmp_path, ev.per_matrix, confusion_table(ev.per_matrix, ev.candidate_kernels),
                       res.selector.feature_importances(), None, n_boot=50)


# ------------------------------------------------------------------ #
# config                                                               #
# ------------------------------------------------------------------ #

class TestConfig:
    def test_default_config_loads_and_maps_to_train_config(self):
        cfg = load_config(DEFAULT_CONFIG_PATH)
        tc = train_config(cfg)
        assert tc.refit_on_train_val is False and tc.tie_tolerance == 0.01
        assert len(tc.feature_columns) == 7 and tc.grid["max_depth"] == [None, 4, 8]

    def _write(self, tmp_path, mutate):
        cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
        mutate(cfg)
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return p

    def test_unknown_keys_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown top-level"):
            load_config(self._write(tmp_path, lambda c: c.update(bogus=1)))
        with pytest.raises(ConfigError, match="unknown key"):
            load_config(self._write(tmp_path, lambda c: c["selector"].update(sed=1)))

    def test_bad_values_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="tie_tolerance"):
            load_config(self._write(tmp_path, lambda c: c["selector"].update(tie_tolerance=-1)))
        with pytest.raises(ConfigError, match="refit_on_train_val"):
            load_config(self._write(tmp_path, lambda c: c["selector"].update(refit_on_train_val="yes")))
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_overrides(self):
        cfg = load_config(DEFAULT_CONFIG_PATH)
        out = apply_overrides(cfg, {"timings": "x.csv", "features": None, "results_dir": "r"})
        assert out["paths"]["timings"] == "x.csv" and out["results_dir"] == "r"
        assert out["paths"]["features"] == cfg["paths"]["features"]       # None -> unchanged
        assert cfg["paths"]["timings"] != "x.csv"                          # original untouched
        with pytest.raises(ConfigError):
            apply_overrides(cfg, {"nope": "1"})

    def test_is_synthetic_and_run_id(self):
        assert is_synthetic({"t": Path("a/SYNTHETIC_kernel_timings.csv")})
        assert not is_synthetic({"t": Path("synthetic_dir/kernel_timings.csv")})   # file NAME only
        assert len(make_run_id().split("-")) == 3

    def test_latest_run_dir(self, tmp_path):
        with pytest.raises(ConfigError, match="no trained run"):
            latest_run_dir(tmp_path)
        for name in ("20260101-0000-a", "20260102-0000-b"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "model.joblib").write_bytes(b"x")
        (tmp_path / "20260103-0000-c").mkdir()          # no model -> ignored
        assert latest_run_dir(tmp_path).name == "20260102-0000-b"


# ------------------------------------------------------------------ #
# CLI end to end                                                       #
# ------------------------------------------------------------------ #

def run_subprocess(script, *args):
    """A real `python scripts/<script>` process (slow: ~6 s of library imports each)."""
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=ROOT)


@functools.lru_cache(maxsize=None)
def _script_module(script):
    spec = importlib.util.spec_from_file_location("cli_" + script[:-3], ROOT / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(script, *args):
    """Call the script's main() in-process; same arguments, return code and output as the CLI."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = _script_module(script).main(list(args))
        except SystemExit as exc:
            rc = exc.code
    return types.SimpleNamespace(returncode=rc, stdout=out.getvalue(), stderr=err.getvalue())


@pytest.fixture(scope="module")
def cli_env(tmp_path_factory):
    base = tmp_path_factory.mktemp("cli")
    _, paths = dataset_from(base / "data")
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    cfg["selector"]["grid"] = {"n_estimators": [10, 20], "max_depth": [None, 3]}
    cfg["selector"]["seed"] = 4
    cfg["evaluation"]["bootstrap_samples"] = 100
    cfg["paths"] = {k: str(v) for k, v in paths.items()}
    cfg["results_dir"] = str(base / "results")
    cfg_path = base / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return base, cfg_path, paths


class TestCli:
    def test_synthetic_input_refused_without_flag(self, cli_env):
        base, cfg, _ = cli_env
        for script in ("train_selector.py", "evaluate_generalization.py"):
            r = run(script, "--config", str(cfg))
            assert r.returncode == 2 and "synthetic" in r.stderr.lower()

    def test_full_run_produces_every_planned_output(self, cli_env):
        """The one true-subprocess test: proves `python scripts/...` itself works."""
        base, cfg, _ = cli_env
        r = run_subprocess("train_selector.py", "--config", str(cfg), "--run-id", "full", "--allow-synthetic")
        assert r.returncode == 0, r.stderr
        assert "SYNTHETIC DATA - NOT A RESULT" in r.stdout
        run_dir = base / "results" / "full"
        e = run_subprocess("evaluate_generalization.py", "--config", str(cfg), "--run-dir", str(run_dir),
                           "--allow-synthetic")
        assert e.returncode == 0, e.stderr
        assert "SYNTHETIC DATA - NOT A RESULT" in e.stdout
        expected = ["model_card.json", "model.joblib", "tuning_results.csv", "excluded_matrices.csv",
                    "per_matrix_test.csv", "summary_overall.csv", "summary_overall.md",
                    "summary_by_group.csv", "summary_by_group.md", "confusion_matrix.csv",
                    "run_manifest.json", "RESULTS_ARE_SYNTHETIC.txt"]
        for name in expected:
            assert (run_dir / name).is_file(), name
        assert len(list((run_dir / "figures").glob("*.png"))) == 6
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["synthetic_data"] is True and manifest["split_evaluated"] == "test"
        assert manifest["n_matrices_evaluated"] == 2
        assert set(pd.read_csv(run_dir / "per_matrix_test.csv")["provenance"]) == {SYNTHETIC_BANNER}

    def test_second_evaluation_refused_unless_overwrite(self, cli_env):
        base, cfg, _ = cli_env
        run("train_selector.py", "--config", str(cfg), "--run-id", "twice", "--allow-synthetic")
        d = str(base / "results" / "twice")
        args = ("--config", str(cfg), "--run-dir", d, "--allow-synthetic")
        assert run("evaluate_generalization.py", *args).returncode == 0
        again = run("evaluate_generalization.py", *args)
        assert again.returncode == 3 and "already exists" in again.stderr
        assert run("evaluate_generalization.py", *args, "--overwrite").returncode == 0

    def test_changed_input_after_training_is_detected(self, cli_env, tmp_path):
        base, cfg, paths = cli_env
        run("train_selector.py", "--config", str(cfg), "--run-id", "hash", "--allow-synthetic")
        copy = tmp_path / "SYNTHETIC_kernel_timings.csv"
        copy.write_text(paths["timings"].read_text() + "\n")      # same content + a blank line
        r = run("evaluate_generalization.py", "--config", str(cfg), "--run-dir", str(base / "results" / "hash"),
                "--timings", str(copy), "--allow-synthetic")
        assert r.returncode == 4 and "timings" in r.stderr and "changed since training" in r.stderr

    def test_training_is_reproducible(self, cli_env):
        base, cfg, _ = cli_env
        for rid in ("rep1", "rep2"):
            assert run("train_selector.py", "--config", str(cfg), "--run-id", rid, "--allow-synthetic").returncode == 0
        a, b = (base / "results" / r for r in ("rep1", "rep2"))
        assert (a / "model_card.json").read_text() == (b / "model_card.json").read_text()
        assert (a / "tuning_results.csv").read_text() == (b / "tuning_results.csv").read_text()
        X = pd.read_csv(base / "data" / "features.csv")
        sa, sb = KernelSelector.load(a / "model.joblib"), KernelSelector.load(b / "model.joblib")
        assert sa.predict(X).tolist() == sb.predict(X).tolist()

    @pytest.mark.parametrize("name,corrupt,message", [
        ("dup_id", lambda t: t.update(features=pd.concat([t["features"], t["features"].iloc[[0]]])),
         "duplicate matrix_id"),
        ("bad_split", lambda t: t["splits"].__setitem__("split", t["splits"]["split"].replace("val", "validation")),
         "invalid split label"),
    ])
    def test_corrupt_inputs_give_clear_errors(self, cli_env, tmp_path, name, corrupt, message):
        base, cfg, _ = cli_env
        t = make_tables()
        t["timings"] = make_timings(t["features"])
        corrupt(t)
        p = write_files(tmp_path, t)
        r = run("train_selector.py", "--config", str(cfg), "--allow-synthetic", "--run-id", name,
                "--features", str(p["features"]), "--splits", str(p["splits"]),
                "--catalog", str(p["catalog"]), "--timings", str(p["timings"]))
        assert r.returncode == 1 and message in r.stderr and "Traceback" not in r.stderr

    def test_missing_kernel_row_is_reported_as_exclusion(self, cli_env, tmp_path):
        base, cfg, _ = cli_env
        t = make_tables()
        tm = make_timings(t["features"])
        t["timings"] = tm[~((tm.matrix_id == "m02") & (tm.kernel_name == "kernel_B"))]
        p = write_files(tmp_path, t)
        r = run("train_selector.py", "--config", str(cfg), "--allow-synthetic", "--run-id", "missing_row",
                "--features", str(p["features"]), "--splits", str(p["splits"]),
                "--catalog", str(p["catalog"]), "--timings", str(p["timings"]))
        assert r.returncode == 0, r.stderr
        excl = pd.read_csv(base / "results" / "missing_row" / "train_excluded_matrices.csv")
        assert excl["matrix_id"].tolist() == ["m02"] and excl["reason"].tolist() == ["missing_candidate_timings"]

    def test_real_header_only_timings_fail_with_a_clear_message(self, cli_env):
        base, cfg, _ = cli_env
        real = ROOT / "benchmarks" / "kernel_timings.csv"
        if len(real.read_text().strip().splitlines()) > 1:
            pytest.skip("real timings delivered")
        r = run("train_selector.py", "--results-dir", str(base / "r_real"))
        assert r.returncode == 1 and "no rows" in r.stderr
