"""
evaluate.py
===========
Build the tidy per-matrix result table for one split.

One row per (matrix, method). ``method`` is one of:
  ``selector``        the trained model's choice
  ``fixed_<kernel>``  always use that kernel (one method per candidate kernel)
  ``oracle``          the measured fastest correct kernel (an upper bound)

Every method is scored by the same function (``score_selections``), so they
are directly comparable. The selector's per-matrix inference time is measured
on this machine's CPU only; it is NOT comparable to GPU time and does not
include feature extraction (Member 2's step).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from evaluation.metrics import SCORE_COLUMNS, score_selections
from selector.baselines import best_fixed_on_train, fixed_selections
from selector.data import Dataset
from selector.labels import DEFAULT_TIE_TOLERANCE, oracle_table
from selector.model import KernelSelector

SELECTOR = "selector"
ORACLE = "oracle"

PER_MATRIX_COLUMNS = [
    "matrix_id", "matrix_group", "split", "method",
    "selected_kernel", "oracle_kernel",
    "selected_kernel_time_ms", "oracle_time_ms", "oracle_relative_loss",
    "selection_correct", "selected_correct_flag",
    "selected_end_to_end_ms", "oracle_end_to_end_ms", "end_to_end_relative_loss",
    "selection_overhead_ms",
]
EXCLUDED_COLUMNS = ["matrix_id", "split", "reason", "detail"]


def fixed_method(kernel: str) -> str:
    return f"fixed_{kernel}"


@dataclass
class EvalResult:
    per_matrix: pd.DataFrame
    excluded: pd.DataFrame
    best_fixed_kernel: Optional[str]
    candidate_kernels: List[str]
    split: str


def method_order(per_matrix: pd.DataFrame, best_fixed_kernel: Optional[str]) -> List[str]:
    """selector, best fixed, other fixed (alphabetical), oracle."""
    methods = set(per_matrix["method"])
    fixed = sorted(m for m in methods if m.startswith("fixed_"))
    best = fixed_method(best_fixed_kernel) if best_fixed_kernel else None
    ordered = [SELECTOR] if SELECTOR in methods else []
    if best in fixed:
        ordered.append(best)
        fixed.remove(best)
    return ordered + fixed + ([ORACLE] if ORACLE in methods else [])


def evaluate(
    selector: KernelSelector,
    dataset: Dataset,
    split: str = "test",
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> EvalResult:
    """Score the selector, every fixed-kernel baseline and the oracle on ``split``.

    The best fixed kernel is chosen from the TRAIN split only.
    Matrices with no correct kernel (no oracle) are excluded and reported.
    """
    unknown = set(selector.classes_) - set(dataset.candidate_kernels)
    if unknown:
        raise ValueError(f"model predicts kernel(s) not in this dataset: {sorted(unknown)}")

    timings = dataset.timings_for(split)
    oracle, no_oracle = oracle_table(timings, tie_tolerance)
    feats = dataset.features_for(split)
    group = dict(zip(feats["matrix_id"], feats["matrix_group"]))
    feats = feats[feats["matrix_id"].isin(set(oracle["matrix_id"]))].reset_index(drop=True)

    excluded = pd.concat(
        [
            dataset.excluded[dataset.excluded["split"] == split][EXCLUDED_COLUMNS],
            no_oracle.assign(split=split)[EXCLUDED_COLUMNS],
        ],
        ignore_index=True,
    ).sort_values("matrix_id").reset_index(drop=True)

    try:
        best_fixed, _ = best_fixed_on_train(dataset)
    except ValueError:
        best_fixed = None

    # Selector: predict one matrix at a time to time the per-matrix inference cost.
    picks, overhead = [], []
    for i in range(len(feats)):
        row = feats.iloc[[i]]
        t0 = time.perf_counter()
        picks.append(str(selector.predict(row)[0]))
        overhead.append((time.perf_counter() - t0) * 1000.0)
    sel_picks = pd.DataFrame({"matrix_id": feats["matrix_id"], "selected_kernel": picks})

    ids = feats["matrix_id"].tolist()
    parts = []

    def add(method: str, selections: pd.DataFrame, overhead_ms=None) -> None:
        scores = score_selections(selections, timings, oracle, tie_tolerance)
        scores.insert(0, "method", method)
        scores["selection_overhead_ms"] = np.nan
        if overhead_ms is not None:
            scores["selection_overhead_ms"] = scores["matrix_id"].map(overhead_ms)
        parts.append(scores)

    add(SELECTOR, sel_picks, dict(zip(feats["matrix_id"], overhead)))
    for kernel in dataset.candidate_kernels:
        add(fixed_method(kernel), fixed_selections(ids, kernel))
    add(ORACLE, oracle[oracle["matrix_id"].isin(ids)][["matrix_id", "oracle_kernel"]]
        .rename(columns={"oracle_kernel": "selected_kernel"}))

    per = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["method"] + SCORE_COLUMNS)
    per["matrix_group"] = per["matrix_id"].map(group)
    per["split"] = split
    per = per[PER_MATRIX_COLUMNS]
    order = {m: i for i, m in enumerate(method_order(per, best_fixed))}
    per = (
        per.assign(_o=per["method"].map(order))
        .sort_values(["_o", "matrix_id"]).drop(columns="_o").reset_index(drop=True)
    )
    return EvalResult(per, excluded, best_fixed, list(dataset.candidate_kernels), split)
