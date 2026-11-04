"""
metrics.py
==========
Pure metric functions for kernel-selection quality.

Notation: for matrix i, selected kernel s_i, ``t_i(k)`` = kernel_time_ms of
kernel k, and ``T*_i`` = time of the oracle (fastest *correct*) kernel.

* oracle-relative loss   loss_i = t_i(s_i) / T*_i - 1      (0 = optimal, 0.5 = 50 % slower)
* slowdown               t_i(s_i) / T*_i                   (= 1 + loss)
* selection correct      s_i passed correctness AND t_i(s_i) <= T*_i * (1 + tol)
* selection accuracy     mean of selection_correct over all scored matrices

Incorrect selections (Q4, default policy)
-----------------------------------------
If the selector picks a kernel whose ``correctness_pass`` is False for that
matrix, the choice is a failure: ``selection_correct`` is False and the loss is
``NaN`` (undefined; a wrong answer has no meaningful runtime). It is counted
in ``n_incorrect_selection`` instead of being hidden inside the mean. Accuracy
counts it as wrong; loss statistics skip it.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from selector.labels import DEFAULT_TIE_TOLERANCE, check_tie_tolerance

_FP_SLACK = 1e-9

SCORE_COLUMNS = [
    "matrix_id",
    "selected_kernel",
    "oracle_kernel",
    "selected_kernel_time_ms",
    "oracle_time_ms",
    "oracle_relative_loss",
    "selection_correct",
    "selected_correct_flag",
    "selected_end_to_end_ms",
    "oracle_end_to_end_ms",
    "end_to_end_relative_loss",
]


# ------------------------------------------------------------------ #
# Elementary formulas                                                  #
# ------------------------------------------------------------------ #

def oracle_relative_loss(selected_time, oracle_time):
    """``selected / oracle - 1`` (element-wise). NaN in -> NaN out."""
    selected = np.asarray(selected_time, dtype=float)
    oracle = np.asarray(oracle_time, dtype=float)
    if np.any(oracle[~np.isnan(oracle)] <= 0):
        raise ValueError("oracle times must be > 0")
    return selected / oracle - 1.0


def slowdown(loss):
    """Slowdown ratio (selected / oracle) from a loss value."""
    return np.asarray(loss, dtype=float) + 1.0


def is_selection_correct(selected_time, oracle_time, selected_passed, tie_tolerance=DEFAULT_TIE_TOLERANCE):
    """True where the pick passed correctness and is within tolerance of the oracle."""
    tol = check_tie_tolerance(tie_tolerance)
    selected = np.asarray(selected_time, dtype=float)
    oracle = np.asarray(oracle_time, dtype=float)
    within = selected <= oracle * (1 + tol) * (1 + _FP_SLACK)
    return within & np.asarray(selected_passed, dtype=bool)


def selection_accuracy(correct_flags: Iterable[bool]) -> float:
    """Fraction of True flags. Empty input -> NaN (not 0, not an error)."""
    flags = np.asarray(list(correct_flags), dtype=bool)
    return float(flags.mean()) if flags.size else float("nan")


def total_time_ratio(selected_times, oracle_times) -> float:
    """``sum(selected) / sum(oracle)``. Dominated by big matrices, unlike mean loss."""
    s = np.asarray(selected_times, dtype=float)
    o = np.asarray(oracle_times, dtype=float)
    keep = ~(np.isnan(s) | np.isnan(o))
    if not keep.any():
        return float("nan")
    return float(s[keep].sum() / o[keep].sum())


def fraction_within(losses, thresholds: Sequence[float] = (0.0, 0.05, 0.10, 0.25)) -> Dict[float, float]:
    """Fraction of matrices whose loss is <= each threshold (NaN losses excluded)."""
    arr = np.asarray(losses, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {float(t): float("nan") for t in thresholds}
    return {float(t): float((arr <= t + _FP_SLACK).mean()) for t in thresholds}


def bootstrap_ci(
    values,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Percentile bootstrap CI over matrices. Returns ``(low, high)``.

    Resamples the matrices with replacement. With very few matrices the
    interval is wide or degenerate, which is the honest answer.
    NaNs are dropped first; fewer than 2 values -> ``(nan, nan)``.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    if statistic is np.mean:
        stats = arr[idx].mean(axis=1)            # vectorised fast path
    else:
        stats = np.array([statistic(arr[row]) for row in idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def summarize_loss(losses, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean / median / p90 / max of loss plus a 95 % bootstrap CI of the mean.

    ``n`` is the number of matrices with a defined loss. Empty -> all NaN, n=0.
    """
    arr = np.asarray(losses, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        nan = float("nan")
        return {"n": 0, "mean_loss": nan, "median_loss": nan, "p90_loss": nan,
                "max_loss": nan, "mean_ci_low": nan, "mean_ci_high": nan}
    lo, hi = bootstrap_ci(arr, np.mean, n_boot=n_boot, seed=seed)
    return {
        "n": int(arr.size),
        "mean_loss": float(arr.mean()),
        "median_loss": float(np.median(arr)),
        "p90_loss": float(np.percentile(arr, 90)),
        "max_loss": float(arr.max()),
        "mean_ci_low": lo,
        "mean_ci_high": hi,
    }


# ------------------------------------------------------------------ #
# Scoring a set of selections                                          #
# ------------------------------------------------------------------ #

def score_selections(
    selections: pd.DataFrame,
    timings: pd.DataFrame,
    oracle: pd.DataFrame,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> pd.DataFrame:
    """Score ``selections`` (columns ``matrix_id, selected_kernel``) against the oracle.

    Parameters
    ----------
    timings : validated timings covering (at least) the selected matrices.
    oracle  : output of ``selector.labels.oracle_table`` for the same matrices.

    Raises
    ------
    ValueError for duplicate matrices, matrices without an oracle, or a
    selected kernel that has no timing row for that matrix. Nothing is
    silently dropped.
    """
    if list(selections.columns) != ["matrix_id", "selected_kernel"]:
        raise ValueError("selections must have exactly columns ['matrix_id', 'selected_kernel']")
    # An empty frame can carry float dtype, which cannot merge with string keys.
    selections = selections.astype({"matrix_id": object, "selected_kernel": object})
    if selections["matrix_id"].duplicated().any():
        dup = selections.loc[selections["matrix_id"].duplicated(), "matrix_id"].tolist()
        raise ValueError(f"duplicate matrix_id in selections: {dup}")
    no_oracle = sorted(set(selections["matrix_id"]) - set(oracle["matrix_id"]))
    if no_oracle:
        raise ValueError(f"selections for matrices with no oracle (filter them first): {no_oracle}")

    t = timings[["matrix_id", "kernel_name", "kernel_time_ms", "preprocessing_time_ms", "correctness_pass"]]
    df = selections.merge(
        t, left_on=["matrix_id", "selected_kernel"], right_on=["matrix_id", "kernel_name"],
        how="left", validate="one_to_one",
    )
    absent = df["kernel_name"].isna()
    if absent.any():
        pairs = list(zip(df.loc[absent, "matrix_id"], df.loc[absent, "selected_kernel"]))
        raise ValueError(f"selected kernel has no timing row for (matrix_id, kernel): {pairs}")

    df = df.merge(
        oracle[["matrix_id", "oracle_kernel", "oracle_time_ms", "oracle_e2e_ms"]],
        on="matrix_id", how="left", validate="many_to_one",
    )
    passed = df["correctness_pass"].astype(bool).to_numpy()
    sel_time = df["kernel_time_ms"].to_numpy(dtype=float)
    sel_e2e = (df["kernel_time_ms"] + df["preprocessing_time_ms"]).to_numpy(dtype=float)

    # A wrong-answer pick has no meaningful runtime: loss is NaN (Q4 default).
    loss = np.where(passed, oracle_relative_loss(sel_time, df["oracle_time_ms"]), np.nan)
    e2e_loss = np.where(passed, oracle_relative_loss(sel_e2e, df["oracle_e2e_ms"]), np.nan)

    out = pd.DataFrame(
        {
            "matrix_id": df["matrix_id"],
            "selected_kernel": df["selected_kernel"],
            "oracle_kernel": df["oracle_kernel"],
            "selected_kernel_time_ms": sel_time,
            "oracle_time_ms": df["oracle_time_ms"],
            "oracle_relative_loss": loss,
            "selection_correct": is_selection_correct(sel_time, df["oracle_time_ms"], passed, tie_tolerance),
            "selected_correct_flag": passed,
            "selected_end_to_end_ms": sel_e2e,
            "oracle_end_to_end_ms": df["oracle_e2e_ms"],
            "end_to_end_relative_loss": e2e_loss,
        }
    )[SCORE_COLUMNS]
    return out.sort_values("matrix_id").reset_index(drop=True)


def summarize_scores(scores: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """One-row summary of a ``score_selections`` table."""
    n_total = int(len(scores))
    summary: Dict[str, float] = {
        "n_matrices": n_total,
        "accuracy": selection_accuracy(scores["selection_correct"]),
        "n_incorrect_selection": int((~scores["selected_correct_flag"]).sum()),
    }
    summary.update(summarize_loss(scores["oracle_relative_loss"], n_boot=n_boot, seed=seed))
    ok = scores["selected_correct_flag"]
    summary["total_time_ratio"] = total_time_ratio(
        scores.loc[ok, "selected_kernel_time_ms"], scores.loc[ok, "oracle_time_ms"]
    )
    return summary
