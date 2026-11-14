"""Unit tests for experiment.methods and experiment.overhead (software validation on tiny tables)."""

import numpy as np
import pandas as pd
import pytest

from experiment import methods as M
from experiment import overhead as O


def _timings(rows):
    return pd.DataFrame(rows, columns=["matrix_id", "kernel_name", "kernel_time_ms",
                                       "preprocessing_time_ms", "correctness_pass"])


def _two_kernel_world(n=12):
    """Kernel 'a' wins when mean_row_nnz < 10, kernel 'b' otherwise."""
    feats, rows = [], []
    for i in range(n):
        mean = 2.0 + 3 * i
        feats.append({"matrix_id": f"m{i}", "matrix_group": f"g{i % 4}", "mean_row_nnz": mean, "rows": 100 + i})
        a, b = (1.0, 2.0) if mean < 10 else (2.0, 1.0)
        rows += [(f"m{i}", "a", a, 0.0, True), (f"m{i}", "b", b, 0.5, True)]
    return pd.DataFrame(feats), _timings(rows)


def test_row_length_heuristic_recovers_single_threshold():
    f, t = _two_kernel_world()
    h = M.RowLengthHeuristic().fit(f, t)
    pred = h.predict(f)
    expected = np.where(f["mean_row_nnz"] < 10, "a", "b")
    assert (pred == expected).all()
    assert len(h.thresholds) >= 1 and "->" in h.describe()


def test_row_length_heuristic_ignores_incorrect_kernel():
    f, t = _two_kernel_world()
    t.loc[(t.kernel_name == "a"), "correctness_pass"] = False   # 'a' is never correct
    h = M.RowLengthHeuristic().fit(f, t)
    # an incorrect kernel can never be recommended, even though it is faster on small rows
    assert set(h.predict(f)) == {"b"}


def test_row_length_heuristic_needs_data():
    f, t = _two_kernel_world()
    t["correctness_pass"] = False
    with pytest.raises(ValueError):
        M.RowLengthHeuristic().fit(f, t)


def test_tree_selector_learns_and_predicts_names():
    f, t = _two_kernel_world(16)
    lab = M._labels(t, f, 0.01)
    tree = M.TreeSelector(["mean_row_nnz", "rows"], max_depth=2).fit(lab, lab["label"])
    assert set(tree.predict(f)) <= {"a", "b"}
    assert (tree.predict(f) == lab["label"].to_numpy()).all()


def test_tune_on_validation_never_sees_test_and_picks_lowest_loss():
    f, t = _two_kernel_world(16)
    tr, va = f.iloc[:10], f.iloc[10:]
    best, tab = M.tune_on_validation(
        lambda **p: M.TreeSelector(["mean_row_nnz"], **p), {"max_depth": [1, 3]},
        tr, t[t.matrix_id.isin(tr.matrix_id)], va, t[t.matrix_id.isin(va.matrix_id)])
    assert tab["chosen"].sum() == 1 and len(tab) == 2


def test_group_folds_keep_groups_whole_and_balanced():
    groups = ["A"] * 10 + ["B"] * 6 + ["C"] * 3 + ["D"] * 3 + ["E"] * 2
    folds = M.group_folds(groups, 3, seed=1)
    assert sorted(g for f in folds for g in f) == list("ABCDE")
    assert all(len(f) > 0 for f in folds)


def test_grouped_cv_uses_only_given_rows():
    f, t = _two_kernel_world(16)
    res = M.grouped_cv_loss(lambda: M.TreeSelector(["mean_row_nnz"], max_depth=2),
                            ["mean_row_nnz"], f, t, k=4)
    assert set(res["matrix_id"]) <= set(f["matrix_id"]) and len(res) > 0


# ---------------------------- overhead -------------------------------------------------------
def test_amortised_totals_and_oracle_bound():
    t = _timings([("m", "fast_setup", 2.0, 0.0, True), ("m", "slow_setup", 1.0, 10.0, True)])
    sel_fast = pd.DataFrame({"matrix_id": ["m"], "selected_kernel": ["fast_setup"]})
    sel_slow = pd.DataFrame({"matrix_id": ["m"], "selected_kernel": ["slow_setup"]})
    assert O.method_total(sel_fast, t, 1)["total_ms"] == pytest.approx(2.0)
    assert O.method_total(sel_slow, t, 1)["total_ms"] == pytest.approx(11.0)
    assert O.oracle_total(t, 1, ["m"]) == pytest.approx(2.0)        # oracle at N=1 avoids the setup
    assert O.oracle_total(t, 100, ["m"]) == pytest.approx(110.0)    # at N=100: min(200, 110)
    assert O.break_even(sel_slow, sel_fast, t) == pytest.approx(10.0)  # 10 + N <= 2N  -> N >= 10


def test_selection_overhead_counted_and_break_even_infinite_when_never_faster():
    t = _timings([("m", "a", 1.0, 0.0, True), ("m", "b", 1.0, 0.0, True)])
    a = pd.DataFrame({"matrix_id": ["m"], "selected_kernel": ["a"]})
    b = pd.DataFrame({"matrix_id": ["m"], "selected_kernel": ["b"]})
    assert O.method_total(a, t, 5, {"m": 3.0})["total_ms"] == pytest.approx(8.0)
    assert O.break_even(a, b, t, overhead_a={"m": 3.0}) == float("inf")


def test_incorrect_selected_kernel_not_scored_as_zero():
    t = _timings([("m", "a", 1.0, 0.0, False), ("m", "b", 2.0, 0.0, True)])
    sel = pd.DataFrame({"matrix_id": ["m"], "selected_kernel": ["a"]})
    r = O.method_total(sel, t, 1)
    assert r["n_scored"] == 0 and r["n_incorrect_selection"] == 1 and r["total_ms"] == 0.0


def test_amortised_table_shape():
    t = _timings([("m", "a", 1.0, 0.0, True), ("m", "b", 2.0, 1.0, True)])
    sel = {"sel": pd.DataFrame({"matrix_id": ["m"], "selected_kernel": ["b"]})}
    tab = O.amortised_table(sel, t, {"sel": {"m": 0.5}}, n_values=(1, 10))
    assert list(tab["n_iterations"]) == [1, 10] and (tab["ratio_to_oracle"] >= 1).all()


# ---------------------------- tie tolerance ------------------------------------------------
def _summary(med, iqr_frac):
    rows = []
    for i, m in enumerate(med):
        rows.append({"matrix_id": f"m{i}", "kernel_name": "a", "median_ms": m, "iqr_ms": m * iqr_frac,
                     "correct_first": True, "correct_last": True})
        rows.append({"matrix_id": f"m{i}", "kernel_name": "b", "median_ms": m * 2, "iqr_ms": m * iqr_frac,
                     "correct_first": True, "correct_last": True})
    return pd.DataFrame(rows)


def test_tie_tolerance_uses_run_to_run_difference_when_repeat_available():
    from experiment.analysis import measure_tie_tolerance
    s1 = _summary([1.0] * 10, 0.20)          # large single-repetition IQR (20 %)
    s2 = _summary([1.03] * 10, 0.20)         # but medians agree within 3 %
    ids = [f"m{i}" for i in range(10)]
    t = measure_tie_tolerance(s1, ids, summary_repeat=s2)
    assert t["tie_tolerance"] == pytest.approx(0.03)
    assert "run-to-run" in t["method"]
    assert t["median_relative_iqr_of_winner"] == pytest.approx(0.20)


def test_tie_tolerance_fallback_and_floor():
    from experiment.analysis import measure_tie_tolerance
    ids = ["m0", "m1"]
    t = measure_tie_tolerance(_summary([1.0, 1.0], 0.0), ids)
    assert t["tie_tolerance"] == pytest.approx(0.01) and "fallback" in t["method"]


def test_tie_tolerance_ignores_non_train_val_matrices():
    from experiment.analysis import measure_tie_tolerance
    s1, s2 = _summary([1.0, 1.0], 0.0), _summary([1.0, 5.0], 0.0)   # m1 differs hugely but is not train/val
    assert measure_tie_tolerance(s1, ["m0"], summary_repeat=s2)["tie_tolerance"] == pytest.approx(0.01)
