"""Tests for selector/data.py (Milestone 1). All data is built in tmp_path."""

import pandas as pd
import pytest

from selector.data import (
    DataContractError,
    FEATURE_COLUMNS,
    build_dataset,
    find_incomplete_matrices,
    load_catalog,
    load_features,
    load_splits,
    load_timings,
)
from tests.fixtures.synthetic_timings import (
    SYNTHETIC_KERNELS,
    make_tables,
    make_timings,
    write_files,
)


def _paths(tmp_path, **overrides):
    return write_files(tmp_path, overrides)


def _build(paths, **kw):
    return build_dataset(paths["features"], paths["splits"], paths["catalog"], paths["timings"], **kw)


def _tables():
    t = make_tables()
    t["timings"] = make_timings(t["features"])
    return t


class TestHappyPath:
    def test_builds_and_shapes(self, tmp_path):
        ds = _build(_paths(tmp_path))
        assert len(ds.features) == 8
        assert ds.candidate_kernels == sorted(SYNTHETIC_KERNELS)
        assert ds.excluded.empty
        assert list(ds.features.columns[:3]) == ["matrix_id", "split", "matrix_group"]
        assert list(ds.features.columns[3:]) == FEATURE_COLUMNS

    def test_split_views(self, tmp_path):
        ds = _build(_paths(tmp_path))
        assert ds.matrix_ids("train") == ["m01", "m02", "m03", "m04"]
        assert ds.matrix_ids("val") == ["m05", "m06"]
        assert set(ds.timings_for("test")["matrix_id"]) == {"m07", "m08"}

    def test_unknown_split_name_rejected(self, tmp_path):
        ds = _build(_paths(tmp_path))
        with pytest.raises(ValueError):
            ds.timings_for("validation")


class TestRowOrderIndependence:
    def test_shuffled_rows_give_identical_dataset(self, tmp_path):
        t = _tables()
        a = _build(write_files(tmp_path / "a", t))
        shuffled = {k: v.sample(frac=1.0, random_state=7).reset_index(drop=True) for k, v in t.items()}
        b = _build(write_files(tmp_path / "b", shuffled))
        pd.testing.assert_frame_equal(a.features, b.features)
        pd.testing.assert_frame_equal(a.timings_for("train"), b.timings_for("train"))
        pd.testing.assert_frame_equal(a.timings_for("test"), b.timings_for("test"))


class TestHeaders:
    @pytest.mark.parametrize("key,loader", [
        ("features", load_features), ("splits", load_splits),
        ("catalog", load_catalog), ("timings", load_timings),
    ])
    def test_wrong_header(self, tmp_path, key, loader):
        t = _tables()
        t[key] = t[key].rename(columns={"matrix_id": "id"})
        with pytest.raises(DataContractError, match="header mismatch"):
            loader(_paths(tmp_path, **t)[key])

    def test_missing_file(self, tmp_path):
        with pytest.raises(DataContractError, match="not found"):
            load_features(tmp_path / "nope.csv")


class TestDuplicatesAndMissing:
    def test_duplicate_matrix_id_in_features(self, tmp_path):
        t = _tables()
        t["features"] = pd.concat([t["features"], t["features"].iloc[[0]]])
        with pytest.raises(DataContractError, match="duplicate matrix_id.*m01"):
            load_features(_paths(tmp_path, **t)["features"])

    def test_duplicate_matrix_id_in_splits_and_catalog(self, tmp_path):
        t = _tables()
        t["splits"] = pd.concat([t["splits"], t["splits"].iloc[[1]]])
        t["catalog"] = pd.concat([t["catalog"], t["catalog"].iloc[[2]]])
        p = _paths(tmp_path, **t)
        with pytest.raises(DataContractError, match="m02"):
            load_splits(p["splits"])
        with pytest.raises(DataContractError, match="m03"):
            load_catalog(p["catalog"])

    def test_duplicate_timing_pair_is_error(self, tmp_path):
        t = _tables()
        t["timings"] = pd.concat([t["timings"], t["timings"].iloc[[0]]])
        with pytest.raises(DataContractError, match="duplicate matrix_id\\+kernel_name"):
            load_timings(_paths(tmp_path, **t)["timings"])

    def test_missing_value(self, tmp_path):
        t = _tables()
        t["features"]["nnz"] = t["features"]["nnz"].astype(object)
        t["features"].loc[2, "nnz"] = ""
        with pytest.raises(DataContractError, match="missing value in column 'nnz'.*m03"):
            load_features(_paths(tmp_path, **t)["features"])

    def test_whitespace_in_matrix_id(self, tmp_path):
        t = _tables()
        t["splits"].loc[0, "matrix_id"] = " m01"
        with pytest.raises(DataContractError, match="whitespace"):
            load_splits(_paths(tmp_path, **t)["splits"])


class TestValueChecks:
    def test_bad_split_label_validation(self, tmp_path):
        t = _tables()
        t["splits"].loc[4, "split"] = "validation"
        with pytest.raises(DataContractError, match="invalid split label.*m05"):
            load_splits(_paths(tmp_path, **t)["splits"])

    def test_nan_and_inf_in_features(self, tmp_path):
        for bad in ("nan", "inf", "abc"):
            t = _tables()
            t["features"]["variance_row_nnz"] = t["features"]["variance_row_nnz"].astype(object)
            t["features"].loc[1, "variance_row_nnz"] = bad
            with pytest.raises(DataContractError, match="m02"):
                load_features(_paths(tmp_path / bad, **t)["features"])

    def test_negative_feature(self, tmp_path):
        t = _tables()
        t["features"].loc[0, "mean_row_nnz"] = -1.0
        with pytest.raises(DataContractError, match=">= 0.*m01"):
            load_features(_paths(tmp_path, **t)["features"])

    def test_non_integer_in_integer_column(self, tmp_path):
        t = _tables()
        t["features"]["rows"] = t["features"]["rows"].astype(float)
        t["features"].loc[0, "rows"] = 10.5
        with pytest.raises(DataContractError, match="must be integer.*m01"):
            load_features(_paths(tmp_path, **t)["features"])

    def test_min_greater_than_max(self, tmp_path):
        t = _tables()
        t["features"].loc[0, ["min_row_nnz", "max_row_nnz"]] = [9, 2]
        with pytest.raises(DataContractError, match="min_row_nnz > max_row_nnz.*m01"):
            load_features(_paths(tmp_path, **t)["features"])

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_kernel_time_must_be_positive(self, tmp_path, value):
        t = _tables()
        t["timings"].loc[0, "kernel_time_ms"] = value
        with pytest.raises(DataContractError, match="kernel_time_ms must be > 0"):
            load_timings(_paths(tmp_path, **t)["timings"])

    def test_negative_preprocessing_time(self, tmp_path):
        t = _tables()
        t["timings"].loc[0, "preprocessing_time_ms"] = -0.1
        with pytest.raises(DataContractError, match="preprocessing_time_ms must be >= 0"):
            load_timings(_paths(tmp_path, **t)["timings"])


class TestCorrectnessParsing:
    @pytest.mark.parametrize("text,expected", [
        ("True", True), ("true", True), ("TRUE", True),
        ("False", False), ("false", False), ("FALSE", False),
    ])
    def test_accepted_spellings(self, tmp_path, text, expected):
        t = _tables()
        t["timings"]["correctness_pass"] = t["timings"]["correctness_pass"].astype(object)
        t["timings"].loc[0, "correctness_pass"] = text
        df = load_timings(_paths(tmp_path, **t)["timings"])
        row = df[(df.matrix_id == t["timings"].loc[0, "matrix_id"])
                 & (df.kernel_name == t["timings"].loc[0, "kernel_name"])]
        assert bool(row["correctness_pass"].iloc[0]) is expected

    @pytest.mark.parametrize("text", ["1", "0", "yes", "pass", "T"])
    def test_rejected_spellings(self, tmp_path, text):
        t = _tables()
        t["timings"]["correctness_pass"] = t["timings"]["correctness_pass"].astype(object)
        t["timings"].loc[0, "correctness_pass"] = text
        with pytest.raises(DataContractError, match="must be True/False"):
            load_timings(_paths(tmp_path / text, **t)["timings"])


class TestCrossFile:
    def test_matrix_id_sets_differ(self, tmp_path):
        t = _tables()
        t["splits"] = t["splits"][t["splits"].matrix_id != "m08"]
        with pytest.raises(DataContractError, match="only in features.*m08"):
            _build(_paths(tmp_path, **t))

    def test_catalog_has_extra_matrix(self, tmp_path):
        t = _tables()
        extra = t["catalog"].iloc[[0]].copy()
        extra["matrix_id"] = "zzz"
        t["catalog"] = pd.concat([t["catalog"], extra])
        with pytest.raises(DataContractError, match="only in catalog.*zzz"):
            _build(_paths(tmp_path, **t))

    def test_timings_for_unknown_matrix(self, tmp_path):
        t = _tables()
        extra = t["timings"].iloc[[0]].copy()
        extra["matrix_id"] = "ghost"
        t["timings"] = pd.concat([t["timings"], extra])
        with pytest.raises(DataContractError, match="absent from.*ghost"):
            _build(_paths(tmp_path, **t))

    def test_features_vs_catalog_disagree(self, tmp_path):
        t = _tables()
        t["catalog"].loc[3, "nnz"] += 1
        with pytest.raises(DataContractError, match="disagree on 'nnz'.*m04"):
            _build(_paths(tmp_path, **t))


class TestCandidateCoverage:
    def test_missing_kernel_row_excludes_matrix(self, tmp_path):
        t = _tables()
        drop = (t["timings"].matrix_id == "m06") & (t["timings"].kernel_name == "kernel_B")
        t["timings"] = t["timings"][~drop]
        ds = _build(_paths(tmp_path, **t))
        assert "m06" not in set(ds.features["matrix_id"])
        assert "m06" not in set(ds.timings_for("val")["matrix_id"])
        row = ds.excluded.iloc[0]
        assert (row.matrix_id, row.split, row.reason, row.detail) == (
            "m06", "val", "missing_candidate_timings", "kernel_B")
        assert len(ds.features) == 7

    def test_matrix_with_no_timings_at_all_is_excluded(self, tmp_path):
        t = _tables()
        t["timings"] = t["timings"][t["timings"].matrix_id != "m01"]
        ds = _build(_paths(tmp_path, **t))
        assert ds.excluded.iloc[0].detail == ";".join(sorted(SYNTHETIC_KERNELS))

    def test_incorrect_kernel_row_is_not_missing(self, tmp_path):
        t = _tables()
        t["timings"].loc[0, "correctness_pass"] = False
        ds = _build(_paths(tmp_path, **t))
        assert ds.excluded.empty

    def test_explicit_candidate_subset(self, tmp_path):
        t = _tables()
        drop = (t["timings"].matrix_id == "m02") & (t["timings"].kernel_name == "kernel_C")
        t["timings"] = t["timings"][~drop]
        ds = _build(_paths(tmp_path, **t), candidate_kernels=["kernel_A", "kernel_B"])
        assert ds.candidate_kernels == ["kernel_A", "kernel_B"]
        assert ds.excluded.empty
        assert set(ds.timings_for("train")["kernel_name"]) == {"kernel_A", "kernel_B"}

    def test_unknown_candidate_rejected(self, tmp_path):
        with pytest.raises(DataContractError, match="never appear"):
            _build(_paths(tmp_path), candidate_kernels=["kernel_A", "nonexistent"])

    def test_find_incomplete_helper(self):
        t = make_timings(make_tables()["features"])
        t = t[~((t.matrix_id == "m01") & (t.kernel_name == "kernel_A"))]
        assert find_incomplete_matrices(["m01", "m02"], t, SYNTHETIC_KERNELS) == {"m01": ["kernel_A"]}


class TestEmptyTimings:
    def test_header_only_loads_but_build_refuses(self, tmp_path):
        t = _tables()
        t["timings"] = t["timings"].iloc[0:0]
        p = _paths(tmp_path, **t)
        assert load_timings(p["timings"]).empty
        with pytest.raises(DataContractError, match="no rows"):
            _build(p)

    def test_real_repo_files_fail_clearly_until_member1_delivers(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        timings = root / "benchmarks" / "kernel_timings.csv"
        if len(load_timings(timings)) > 0:
            pytest.skip("real timings delivered; this guard no longer applies")
        with pytest.raises(DataContractError, match="no rows"):
            build_dataset(root / "data/metadata/features.csv", root / "data/metadata/splits.csv",
                          root / "data/metadata/matrix_catalog.csv", timings)


class TestRealMetadataFiles:
    def test_pilot_metadata_files_are_valid(self):
        from pathlib import Path
        meta = Path(__file__).resolve().parents[1] / "data" / "metadata"
        assert len(load_features(meta / "features.csv")) == 7
        assert set(load_splits(meta / "splits.csv")["split"]) == {"train", "val", "test"}
        assert len(load_catalog(meta / "matrix_catalog.csv")) == 7
