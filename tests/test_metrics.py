"""Tests for evaluation/metrics.py. Expected values are computed by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    SCORE_COLUMNS,
    bootstrap_ci,
    fraction_within,
    is_selection_correct,
    oracle_relative_loss,
    score_selections,
    selection_accuracy,
    slowdown,
    summarize_loss,
    summarize_scores,
    total_time_ratio,
)
from selector.labels import oracle_table


def T(rows):
    return pd.DataFrame(rows, columns=[
        "matrix_id", "kernel_name", "kernel_time_ms", "preprocessing_time_ms", "correctness_pass"])


def sel(pairs):
    return pd.DataFrame(pairs, columns=["matrix_id", "selected_kernel"])


# Oracle (K1) times: a=2, b=4, c=10. K2 times: 3, 6, 11.
TIMINGS = T([
    ("a", "K1", 2.0, 0.0, True), ("a", "K2", 3.0, 0.0, True),
    ("b", "K1", 4.0, 0.0, True), ("b", "K2", 6.0, 0.0, True),
    ("c", "K1", 10.0, 0.0, True), ("c", "K2", 11.0, 0.0, True),
])


class TestLossFormula:
    def test_zero_when_selector_picks_oracle(self):
        assert oracle_relative_loss(2.0, 2.0) == 0.0

    def test_half_for_one_and_half_times_slower(self):
        assert oracle_relative_loss(6.0, 4.0) == pytest.approx(0.5)

    def test_vectorised_and_nan_passthrough(self):
        out = oracle_relative_loss([2.0, 6.0, np.nan], [2.0, 4.0, 5.0])
        assert out[:2] == pytest.approx([0.0, 0.5]) and math.isnan(out[2])

    def test_nonpositive_oracle_rejected(self):
        with pytest.raises(ValueError):
            oracle_relative_loss(1.0, 0.0)

    def test_slowdown_is_one_plus_loss(self):
        assert slowdown(0.5) == pytest.approx(1.5)


class TestSelectionCorrect:
    def test_exact_pick(self):
        assert is_selection_correct(2.0, 2.0, True)

    def test_within_tolerance_counts_as_correct(self):
        assert is_selection_correct(2.01, 2.0, True, tie_tolerance=0.01)  # exactly +0.5 %

    def test_outside_tolerance_is_wrong(self):
        assert not is_selection_correct(2.1, 2.0, True, tie_tolerance=0.01)

    def test_boundary_one_percent(self):
        assert is_selection_correct(1.01, 1.0, True, tie_tolerance=0.01)

    def test_failed_correctness_is_never_correct(self):
        assert not is_selection_correct(1.0, 2.0, False)


class TestAccuracy:
    def test_basic(self):
        assert selection_accuracy([True, False, True, True]) == 0.75

    def test_empty_is_nan(self):
        assert math.isnan(selection_accuracy([]))


class TestAggregates:
    def test_hand_computed_summary(self):
        s = summarize_loss([0.0, 0.1, 0.5])
        assert s["n"] == 3
        assert s["mean_loss"] == pytest.approx(0.2)
        assert s["median_loss"] == pytest.approx(0.1)
        assert s["p90_loss"] == pytest.approx(0.42)   # 0.1 + 0.8 * (0.5 - 0.1)
        assert s["max_loss"] == pytest.approx(0.5)

    def test_mean_vs_median_differ_with_outlier(self):
        s = summarize_loss([0.0, 0.0, 0.0, 4.0])
        assert s["mean_loss"] == pytest.approx(1.0) and s["median_loss"] == 0.0

    def test_empty_set(self):
        s = summarize_loss([])
        assert s["n"] == 0 and math.isnan(s["mean_loss"]) and math.isnan(s["mean_ci_low"])

    def test_nan_losses_skipped_and_not_counted(self):
        s = summarize_loss([0.0, np.nan, 0.5])
        assert s["n"] == 2 and s["mean_loss"] == pytest.approx(0.25)

    def test_total_time_ratio(self):
        assert total_time_ratio([2, 6, 11], [2, 4, 10]) == pytest.approx(19 / 16)

    def test_total_time_ratio_skips_nan_pairs(self):
        assert total_time_ratio([2, np.nan], [2, 4]) == pytest.approx(1.0)

    def test_fraction_within(self):
        f = fraction_within([0.0, 0.04, 0.2, 0.6], thresholds=(0.0, 0.05, 0.25))
        assert f == {0.0: 0.25, 0.05: 0.5, 0.25: 0.75}

    def test_fraction_within_empty(self):
        assert math.isnan(fraction_within([])[0.0])


class TestBootstrap:
    def test_deterministic_for_same_seed(self):
        v = [0.0, 0.1, 0.5, 0.2, 0.05]
        assert bootstrap_ci(v, seed=1) == bootstrap_ci(v, seed=1)

    def test_interval_contains_mean_and_is_ordered(self):
        v = [0.0, 0.1, 0.5, 0.2, 0.05, 0.3]
        lo, hi = bootstrap_ci(v)
        assert lo <= np.mean(v) <= hi

    def test_constant_data_gives_degenerate_interval(self):
        assert bootstrap_ci([0.2, 0.2, 0.2]) == (pytest.approx(0.2), pytest.approx(0.2))

    def test_too_few_values_is_nan(self):
        lo, hi = bootstrap_ci([0.3])
        assert math.isnan(lo) and math.isnan(hi)


class TestScoreSelections:
    def _score(self, pairs, timings=TIMINGS, **kw):
        oracle, _ = oracle_table(timings)
        return score_selections(sel(pairs), timings, oracle, **kw)

    def test_hand_computed_table(self):
        s = self._score([("a", "K1"), ("b", "K2"), ("c", "K2")])
        assert list(s.columns) == SCORE_COLUMNS
        assert s["oracle_relative_loss"].tolist() == pytest.approx([0.0, 0.5, 0.1])
        assert s["selection_correct"].tolist() == [True, False, False]
        assert s["selected_kernel_time_ms"].tolist() == [2.0, 6.0, 11.0]
        assert s["oracle_time_ms"].tolist() == [2.0, 4.0, 10.0]

    def test_summary_matches_hand_values(self):
        summ = summarize_scores(self._score([("a", "K1"), ("b", "K2"), ("c", "K2")]))
        assert summ["n_matrices"] == 3 and summ["n_incorrect_selection"] == 0
        assert summ["accuracy"] == pytest.approx(1 / 3)
        assert summ["mean_loss"] == pytest.approx(0.2)
        assert summ["median_loss"] == pytest.approx(0.1)
        assert summ["total_time_ratio"] == pytest.approx(19 / 16)

    def test_perfect_selector(self):
        summ = summarize_scores(self._score([("a", "K1"), ("b", "K1"), ("c", "K1")]))
        assert summ["accuracy"] == 1.0 and summ["mean_loss"] == 0.0 and summ["max_loss"] == 0.0

    def test_near_tie_pick_is_correct_with_nonzero_loss(self):
        t = T([("a", "K1", 1.000, 0, True), ("a", "K2", 1.005, 0, True)])
        s = self._score([("a", "K2")], timings=t)
        assert s["selection_correct"].iloc[0]
        assert s["oracle_relative_loss"].iloc[0] == pytest.approx(0.005)

    def test_selecting_incorrect_kernel_is_failure_with_nan_loss(self):
        t = T([("a", "K1", 2.0, 0, True), ("a", "K2", 1.0, 0, False),
               ("b", "K1", 4.0, 0, True), ("b", "K2", 5.0, 0, True)])
        s = self._score([("a", "K2"), ("b", "K1")], timings=t)
        row = s.set_index("matrix_id").loc["a"]
        assert not row.selection_correct and not row.selected_correct_flag
        assert math.isnan(row.oracle_relative_loss)
        summ = summarize_scores(s)
        assert summ["n_incorrect_selection"] == 1
        assert summ["accuracy"] == 0.5          # failure counts as wrong
        assert summ["n"] == 1                   # loss stats skip it, count shown separately
        assert summ["mean_loss"] == 0.0

    def test_end_to_end_columns(self):
        t = T([("a", "K1", 1.0, 5.0, True), ("a", "K2", 2.0, 0.0, True)])
        s = self._score([("a", "K1")], timings=t).iloc[0]
        assert s.selected_end_to_end_ms == 6.0 and s.oracle_end_to_end_ms == 2.0
        assert s.end_to_end_relative_loss == pytest.approx(2.0)  # 6 / 2 - 1
        assert s.oracle_relative_loss == 0.0                       # kernel-only is optimal

    def test_row_order_of_selections_irrelevant(self):
        a = self._score([("a", "K1"), ("b", "K2"), ("c", "K2")])
        b = self._score([("c", "K2"), ("a", "K1"), ("b", "K2")])
        pd.testing.assert_frame_equal(a, b)

    def test_empty_selections(self):
        s = self._score([])
        assert s.empty and list(s.columns) == SCORE_COLUMNS
        summ = summarize_scores(s)
        assert summ["n_matrices"] == 0 and math.isnan(summ["accuracy"]) and math.isnan(summ["mean_loss"])


class TestScoreErrors:
    def _oracle(self):
        return oracle_table(TIMINGS)[0]

    def test_duplicate_matrix(self):
        with pytest.raises(ValueError, match="duplicate"):
            score_selections(sel([("a", "K1"), ("a", "K2")]), TIMINGS, self._oracle())

    def test_matrix_without_oracle(self):
        with pytest.raises(ValueError, match="no oracle"):
            score_selections(sel([("zzz", "K1")]), TIMINGS, self._oracle())

    def test_unknown_kernel(self):
        with pytest.raises(ValueError, match="no timing row"):
            score_selections(sel([("a", "K9")]), TIMINGS, self._oracle())

    def test_wrong_columns(self):
        with pytest.raises(ValueError, match="exactly columns"):
            score_selections(pd.DataFrame({"matrix_id": ["a"]}), TIMINGS, self._oracle())
