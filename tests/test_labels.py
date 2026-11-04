"""Tests for selector/labels.py (oracle + training labels)."""

import pandas as pd
import pytest

from selector.data import build_dataset
from selector.labels import oracle_table, training_labels
from tests.fixtures.synthetic_timings import make_tables, make_timings, write_files


def T(rows):
    """rows: (matrix_id, kernel, time, prep, correct)."""
    return pd.DataFrame(rows, columns=[
        "matrix_id", "kernel_name", "kernel_time_ms", "preprocessing_time_ms", "correctness_pass"])


def oracle_row(timings, mid, **kw):
    oracle, _ = oracle_table(timings, **kw)
    return oracle.set_index("matrix_id").loc[mid]


class TestOracle:
    def test_picks_fastest(self):
        t = T([("a", "K1", 3.0, 0, True), ("a", "K2", 2.0, 0, True), ("a", "K3", 5.0, 0, True)])
        r = oracle_row(t, "a")
        assert (r.oracle_kernel, r.oracle_time_ms, r.n_correct_kernels) == ("K2", 2.0, 3)

    def test_ignores_incorrect_kernels(self):
        # K1 is fastest but WRONG -> must never be the oracle.
        t = T([("a", "K1", 1.0, 0, False), ("a", "K2", 2.0, 0, True), ("a", "K3", 3.0, 0, True)])
        r = oracle_row(t, "a")
        assert (r.oracle_kernel, r.oracle_time_ms, r.n_correct_kernels) == ("K2", 2.0, 2)
        assert "K1" not in r.tied_kernels

    def test_no_correct_kernel_is_excluded_and_reported(self):
        t = T([("a", "K1", 1.0, 0, False), ("a", "K2", 2.0, 0, False), ("b", "K1", 1.0, 0, True)])
        oracle, no_oracle = oracle_table(t)
        assert oracle["matrix_id"].tolist() == ["b"]
        assert no_oracle.to_dict("records") == [
            {"matrix_id": "a", "reason": "no_correct_kernel", "detail": "K1;K2"}]

    def test_all_matrices_without_oracle(self):
        oracle, no_oracle = oracle_table(T([("a", "K1", 1.0, 0, False)]))
        assert oracle.empty and len(no_oracle) == 1

    def test_exact_tie_broken_alphabetically(self):
        t = T([("a", "K2", 1.0, 0, True), ("a", "K1", 1.0, 0, True)])
        assert oracle_row(t, "a").oracle_kernel == "K1"

    def test_row_order_does_not_change_result(self):
        t = T([("a", "K2", 1.0, 0, True), ("a", "K1", 1.0, 0, True), ("b", "K1", 2.0, 0, True),
               ("b", "K2", 1.5, 0, True)])
        a, _ = oracle_table(t)
        b, _ = oracle_table(t.sample(frac=1.0, random_state=3))
        pd.testing.assert_frame_equal(a, b)


class TestTieRule:
    def test_tied_kernels_within_tolerance(self):
        t = T([("a", "K1", 1.000, 0, True), ("a", "K2", 1.005, 0, True), ("a", "K3", 1.020, 0, True)])
        r = oracle_row(t, "a", tie_tolerance=0.01)
        assert r.oracle_kernel == "K1"          # label stays the single fastest
        assert r.tied_kernels == "K1;K2"        # K3 is 2 % slower -> not tied

    def test_boundary_exactly_at_tolerance_is_tied(self):
        t = T([("a", "K1", 1.0, 0, True), ("a", "K2", 1.01, 0, True)])
        assert oracle_row(t, "a", tie_tolerance=0.01).tied_kernels == "K1;K2"

    def test_zero_tolerance_means_only_exact_ties(self):
        t = T([("a", "K1", 1.0, 0, True), ("a", "K2", 1.0001, 0, True)])
        assert oracle_row(t, "a", tie_tolerance=0.0).tied_kernels == "K1"

    def test_incorrect_kernel_never_tied(self):
        t = T([("a", "K1", 1.0, 0, True), ("a", "K2", 1.0, 0, False)])
        assert oracle_row(t, "a").tied_kernels == "K1"

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
    def test_invalid_tolerance_rejected(self, bad):
        with pytest.raises(ValueError, match="tie_tolerance"):
            oracle_table(T([("a", "K1", 1.0, 0, True)]), tie_tolerance=bad)


class TestEndToEndOracle:
    def test_e2e_oracle_can_differ_from_kernel_oracle(self):
        # K1: 1.0 + 5.0 = 6.0 ; K2: 2.0 + 0.0 = 2.0
        t = T([("a", "K1", 1.0, 5.0, True), ("a", "K2", 2.0, 0.0, True)])
        r = oracle_row(t, "a")
        assert (r.oracle_kernel, r.oracle_time_ms) == ("K1", 1.0)
        assert (r.oracle_e2e_kernel, r.oracle_e2e_ms) == ("K2", 2.0)

    def test_e2e_ignores_incorrect_kernels(self):
        t = T([("a", "K1", 1.0, 0.0, False), ("a", "K2", 2.0, 1.0, True)])
        assert oracle_row(t, "a").oracle_e2e_ms == 3.0


class TestInputChecks:
    def test_missing_column(self):
        with pytest.raises(ValueError, match="missing columns"):
            oracle_table(T([("a", "K1", 1.0, 0, True)]).drop(columns=["correctness_pass"]))


class TestTrainingLabels:
    def _dataset(self, tmp_path, timings=None):
        t = make_tables()
        t["timings"] = timings if timings is not None else make_timings(t["features"])
        p = write_files(tmp_path, t)
        return build_dataset(p["features"], p["splits"], p["catalog"], p["timings"])

    def test_labels_only_for_train_matrices(self, tmp_path):
        labels, no_oracle = training_labels(self._dataset(tmp_path))
        assert labels["matrix_id"].tolist() == ["m01", "m02", "m03", "m04"]
        assert list(labels.columns) == ["matrix_id", "label"]
        assert no_oracle.empty

    def test_labels_match_oracle_of_train_timings(self, tmp_path):
        ds = self._dataset(tmp_path)
        labels, _ = training_labels(ds)
        oracle, _ = oracle_table(ds.timings_for("train"))
        assert labels["label"].tolist() == oracle["oracle_kernel"].tolist()

    def test_tampering_with_val_and_test_timings_cannot_change_labels(self, tmp_path):
        base = make_timings(make_tables()["features"])
        a, _ = training_labels(self._dataset(tmp_path / "a", base))
        tampered = base.copy()
        held_out = tampered["matrix_id"].isin(["m05", "m06", "m07", "m08"])
        reversed_times = tampered.loc[held_out, "kernel_time_ms"].iloc[::-1].to_numpy()
        tampered.loc[held_out, "kernel_time_ms"] = reversed_times
        tampered.loc[held_out, "correctness_pass"] = False
        b, _ = training_labels(self._dataset(tmp_path / "b", tampered))
        pd.testing.assert_frame_equal(a, b)

    def test_train_matrix_without_correct_kernel_is_reported(self, tmp_path):
        t = make_timings(make_tables()["features"])
        t.loc[t["matrix_id"] == "m02", "correctness_pass"] = False
        labels, no_oracle = training_labels(self._dataset(tmp_path, t))
        assert "m02" not in labels["matrix_id"].tolist()
        assert no_oracle["matrix_id"].tolist() == ["m02"]
