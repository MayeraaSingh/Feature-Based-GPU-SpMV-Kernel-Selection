"""
labels.py
=========
Oracle and training labels, computed from measured kernel timings.

Definitions (see docs/selector_evaluation.md once written, Milestone 5)
---------------------------------------------------------------------
* A kernel is *eligible* for a matrix only if ``correctness_pass`` is True.
  A kernel that computed a wrong answer can never be the oracle or a label.
* The **oracle kernel** for a matrix is the eligible kernel with the lowest
  ``kernel_time_ms``. Exact ties are broken alphabetically by kernel name so
  the result never depends on row order.
* **Tie tolerance**: every eligible kernel within ``tie_tolerance`` (relative,
  default 1 %) of the fastest time is recorded in ``tied_kernels``. Timing
  noise makes smaller differences meaningless, so picking any of them counts
  as a correct selection (used by ``evaluation.metrics``). The *training
  label* is still the single deterministic oracle kernel.
* A separate **end-to-end oracle** minimises ``kernel_time_ms +
  preprocessing_time_ms`` and is reported alongside, never mixed in.
* A matrix with no eligible kernel has no oracle. It is excluded and reported.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from selector.data import Dataset

DEFAULT_TIE_TOLERANCE = 0.01  # 1 % relative; configurable, needs team agreement

ORACLE_COLUMNS = [
    "matrix_id",
    "oracle_kernel",
    "oracle_time_ms",
    "tied_kernels",
    "n_correct_kernels",
    "oracle_e2e_kernel",
    "oracle_e2e_ms",
]
NO_ORACLE_HEADERS = ["matrix_id", "reason", "detail"]

# Relative slack so a value exactly on the tolerance boundary is not lost
# to floating-point rounding (e.g. 1.0 * (1 + 0.01) != 1.01 in binary).
_FP_SLACK = 1e-9


def check_tie_tolerance(tie_tolerance: float) -> float:
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError(f"tie_tolerance must be a finite number >= 0, got {tie_tolerance!r}")
    return float(tie_tolerance)


def oracle_table(
    timings: pd.DataFrame,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the per-matrix oracle from a (validated) timings table.

    Returns
    -------
    oracle : DataFrame with ``ORACLE_COLUMNS``, one row per matrix that has at
        least one correct kernel, sorted by ``matrix_id``.
    no_oracle : DataFrame with ``NO_ORACLE_HEADERS`` listing matrices where no
        kernel passed correctness.
    """
    tol = check_tie_tolerance(tie_tolerance)
    required = {"matrix_id", "kernel_name", "kernel_time_ms", "preprocessing_time_ms", "correctness_pass"}
    missing = required - set(timings.columns)
    if missing:
        raise ValueError(f"timings is missing columns: {sorted(missing)}")

    eligible = timings[timings["correctness_pass"].astype(bool)].copy()

    all_ids = sorted(timings["matrix_id"].unique())
    ok_ids = set(eligible["matrix_id"])
    no_oracle = pd.DataFrame(
        [
            {"matrix_id": m, "reason": "no_correct_kernel",
             "detail": ";".join(sorted(timings.loc[timings["matrix_id"] == m, "kernel_name"]))}
            for m in all_ids if m not in ok_ids
        ],
        columns=NO_ORACLE_HEADERS,
    )
    if eligible.empty:
        return pd.DataFrame(columns=ORACLE_COLUMNS), no_oracle

    # Fastest correct kernel (ties broken by name -> order independent).
    ordered = eligible.sort_values(["matrix_id", "kernel_time_ms", "kernel_name"])
    best = ordered.groupby("matrix_id", sort=True).head(1).set_index("matrix_id")

    # Every correct kernel within tolerance of the fastest.
    eligible = eligible.join(best["kernel_time_ms"].rename("_best"), on="matrix_id")
    near = eligible[eligible["kernel_time_ms"] <= eligible["_best"] * (1 + tol) * (1 + _FP_SLACK)]
    tied = near.groupby("matrix_id")["kernel_name"].agg(lambda s: ";".join(sorted(s)))

    # End-to-end oracle (kernel + preprocessing), reported separately.
    eligible["_e2e"] = eligible["kernel_time_ms"] + eligible["preprocessing_time_ms"]
    best_e2e = (
        eligible.sort_values(["matrix_id", "_e2e", "kernel_name"])
        .groupby("matrix_id", sort=True).head(1).set_index("matrix_id")
    )

    oracle = pd.DataFrame(
        {
            "matrix_id": best.index,
            "oracle_kernel": best["kernel_name"].to_numpy(),
            "oracle_time_ms": best["kernel_time_ms"].to_numpy(),
            "tied_kernels": tied.reindex(best.index).to_numpy(),
            "n_correct_kernels": eligible.groupby("matrix_id").size().reindex(best.index).to_numpy(),
            "oracle_e2e_kernel": best_e2e.reindex(best.index)["kernel_name"].to_numpy(),
            "oracle_e2e_ms": best_e2e.reindex(best.index)["_e2e"].to_numpy(),
        }
    )[ORACLE_COLUMNS]
    return oracle.reset_index(drop=True), no_oracle


def training_labels(
    dataset: Dataset,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Training labels: the oracle kernel of every *train-split* matrix.

    Leakage guard: this function asks the dataset for ``timings_for("train")``
    and nothing else, so validation and test timings cannot influence labels.

    Returns
    -------
    labels : DataFrame ``matrix_id, label`` (label = oracle kernel name)
    no_oracle : train matrices dropped because no kernel was correct
    """
    oracle, no_oracle = oracle_table(dataset.timings_for("train"), tie_tolerance)
    labels = oracle[["matrix_id", "oracle_kernel"]].rename(columns={"oracle_kernel": "label"})
    return labels.reset_index(drop=True), no_oracle
