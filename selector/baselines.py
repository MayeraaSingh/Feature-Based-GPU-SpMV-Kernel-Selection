"""
baselines.py
============
Fixed-kernel baselines: "just always use kernel X".

A selector is only worth having if it beats the best fixed choice, so these
are the fair comparison. Everything here is decided from the **train split
only**; the chosen kernel is then applied unchanged to other splits.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd

from selector.data import Dataset
from selector.labels import DEFAULT_TIE_TOLERANCE, oracle_table, training_labels


def fixed_selections(matrix_ids: Sequence[str], kernel: str) -> pd.DataFrame:
    """A selections table that picks ``kernel`` for every matrix.

    Same format as a model's output (``matrix_id, selected_kernel``), so it
    goes through the exact same scoring code as the selector.
    """
    return pd.DataFrame({"matrix_id": list(matrix_ids), "selected_kernel": kernel}).sort_values(
        "matrix_id"
    ).reset_index(drop=True)


def train_kernel_table(dataset: Dataset) -> pd.DataFrame:
    """Per-kernel totals over train matrices that have an oracle.

    Columns: ``kernel_name, total_time_ms, n_matrices, n_incorrect, eligible``.
    A kernel is *eligible* as the fixed baseline only if it was correct on
    every train matrix (a fixed kernel that returns wrong answers is not a
    usable default).
    """
    train = dataset.timings_for("train")
    oracle, _ = oracle_table(train)
    train = train[train["matrix_id"].isin(set(oracle["matrix_id"]))]
    grouped = train.groupby("kernel_name")
    table = pd.DataFrame(
        {
            "total_time_ms": grouped["kernel_time_ms"].sum(),
            "n_matrices": grouped.size(),
            "n_incorrect": grouped["correctness_pass"].apply(lambda s: int((~s.astype(bool)).sum())),
        }
    ).reset_index()
    table["eligible"] = table["n_incorrect"] == 0
    return table.sort_values("kernel_name").reset_index(drop=True)


def best_fixed_on_train(dataset: Dataset) -> Tuple[str, pd.DataFrame]:
    """The single kernel with the lowest total train time among eligible kernels.

    Ties break alphabetically. Raises ``ValueError`` if no kernel was correct
    on every train matrix.

    Returns ``(kernel_name, per-kernel table)``.
    """
    table = train_kernel_table(dataset)
    eligible = table[table["eligible"]]
    if eligible.empty:
        raise ValueError("no kernel is correct on every train matrix; cannot pick a fixed baseline")
    best = eligible.sort_values(["total_time_ms", "kernel_name"]).iloc[0]["kernel_name"]
    return str(best), table


def majority_kernel_on_train(dataset: Dataset, tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> str:
    """The kernel that is the oracle on the most train matrices (ties: alphabetical).

    This is what a classifier that ignores features would predict.
    """
    labels, _ = training_labels(dataset, tie_tolerance)
    if labels.empty:
        raise ValueError("no train matrices with an oracle label")
    counts = labels["label"].value_counts()
    top = counts[counts == counts.max()].index
    return str(sorted(top)[0])
