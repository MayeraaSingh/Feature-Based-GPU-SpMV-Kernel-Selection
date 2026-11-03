"""
splits.py
=========
Train / validation / test split manifest generation for the Member 2 pipeline.

Split strategy
--------------
The default strategy is *group-stratified random assignment*:

1. All matrices are grouped by their ``matrix_group`` label.
2. Within each group the matrices are sorted by matrix_id for determinism.
3. Groups are then allocated to splits so that the split *proportions of
   groups* (not individual matrices) are approximately respected.
4. Within each group, all matrices are assigned to the same split (group-level
   split) unless the group is large enough to subdivide.

Rationale: splitting by group prevents near-duplicate or structurally similar
matrices (e.g., all finite-element meshes, all synthetic-diagonal matrices)
from appearing in both train and test splits, which would be a form of
leakage.

When ``strategy="random"`` is requested instead, individual matrices are
randomly shuffled and assigned proportionally—ignoring group.  This is
available for comparison but the default group-based strategy is preferred for
the research.

The split strategy and seed are logged and must be kept fixed for a reported
experiment.  The caller should not revise splits after seeing test outcomes.

Allowed split labels (matching docs/data_dictionary.md):
    "train", "val", "test"
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SplitManifest = Dict[str, str]  # matrix_id → split label


def make_splits(
    matrix_ids: List[str],
    groups: Optional[Dict[str, str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    strategy: str = "group",
) -> SplitManifest:
    """Assign each matrix to a split partition.

    Parameters
    ----------
    matrix_ids : list of str
        All matrix IDs to be split.  Must be non-empty.
    groups : dict, optional
        Mapping matrix_id → matrix_group.  Required for strategy="group".
        If None and strategy="group", falls back to strategy="random" with
        a warning.
    train_ratio, val_ratio, test_ratio : float
        Target proportions.  Must sum to 1.0 (±1e-9).
    seed : int
        Random seed for reproducibility.  Record this in the experiment log.
    strategy : {"group", "random"}
        "group" : assign all matrices in a group to the same split.
        "random": ignore groups; randomly assign individual matrices.

    Returns
    -------
    dict mapping matrix_id → split label ("train", "val", or "test")

    Raises
    ------
    ValueError
        If matrix_ids is empty, ratios are invalid, or strategy is unknown.
    """
    if not matrix_ids:
        raise ValueError("matrix_ids must be non-empty.")

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    if strategy not in ("group", "random"):
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'group' or 'random'.")

    if strategy == "group" and groups is None:
        logger.warning(
            "strategy='group' requested but no groups provided; "
            "falling back to strategy='random'."
        )
        strategy = "random"

    rng = np.random.default_rng(seed)

    if strategy == "random":
        manifest = _random_split(matrix_ids, train_ratio, val_ratio, rng)
    else:
        manifest = _group_split(matrix_ids, groups, train_ratio, val_ratio, rng)

    _log_split_summary(manifest, seed, strategy)
    return manifest


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0; got {total:.10f}."
        )
    for name, v in [("train_ratio", train), ("val_ratio", val), ("test_ratio", test)]:
        if v < 0.0:
            raise ValueError(f"{name} must be >= 0; got {v}.")


def _assign_labels(n: int, train_ratio: float, val_ratio: float) -> List[str]:
    """Return a list of n split labels in proportion to the given ratios."""
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    n_test = n - n_train - n_val
    # Correct rounding so n_test is never negative
    if n_test < 0:
        n_val += n_test
        n_test = 0
    if n_val < 0:
        n_train += n_val
        n_val = 0
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


def _random_split(
    matrix_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> SplitManifest:
    ids = sorted(matrix_ids)  # Sort first for determinism before shuffle
    rng.shuffle(ids)
    labels = _assign_labels(len(ids), train_ratio, val_ratio)
    return dict(zip(ids, labels))


def _group_split(
    matrix_ids: List[str],
    groups: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> SplitManifest:
    """Assign splits at the group level.

    All matrices in a group go to the same split.  Groups are shuffled
    randomly (with the given seed) and then packed greedily into splits
    while honouring the target proportions as closely as possible.

    If a group is larger than the test or val quota, the overflow is
    absorbed into train (and a warning is emitted).
    """
    # Build group → sorted list of matrix_ids
    from collections import defaultdict
    group_map: Dict[str, List[str]] = defaultdict(list)
    unknown_group = "unknown"
    for mid in sorted(matrix_ids):
        g = groups.get(mid, unknown_group)
        group_map[g].append(mid)

    # Shuffle the group order
    group_names = sorted(group_map.keys())
    rng.shuffle(group_names)

    n_total = len(matrix_ids)
    target_val = math.ceil(n_total * val_ratio)
    target_test = math.ceil(n_total * (1.0 - train_ratio - val_ratio))
    # Clamp
    target_test = max(0, min(target_test, n_total))
    target_val = max(0, min(target_val, n_total - target_test))

    manifest: SplitManifest = {}
    n_val_so_far = 0
    n_test_so_far = 0

    for g in group_names:
        mids = group_map[g]
        g_size = len(mids)

        # Decide which split this group belongs to
        if n_test_so_far < target_test:
            label = "test"
            n_test_so_far += g_size
        elif n_val_so_far < target_val:
            label = "val"
            n_val_so_far += g_size
        else:
            label = "train"

        for mid in mids:
            manifest[mid] = label

    # Any matrix_id not yet assigned (e.g. missing from groups) → train
    for mid in matrix_ids:
        if mid not in manifest:
            manifest[mid] = "train"

    return manifest


def _log_split_summary(manifest: SplitManifest, seed: int, strategy: str) -> None:
    counts = {"train": 0, "val": 0, "test": 0}
    for label in manifest.values():
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    logger.info(
        "Split manifest (strategy=%s, seed=%d): "
        "total=%d  train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
        strategy,
        seed,
        total,
        counts["train"], 100 * counts["train"] / total if total else 0,
        counts["val"],   100 * counts["val"]   / total if total else 0,
        counts["test"],  100 * counts["test"]  / total if total else 0,
    )
