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
    """Assign splits at the group level (all matrices of a group share a split).

    Size-aware greedy packing: groups are processed from largest to smallest (ties
    in random order, from the seeded generator) and each group goes to the split
    whose resulting size vector is closest (squared error) to the target
    proportions.  The earlier implementation ignored group size, so a single large
    group could land in val/test and leave train tiny (e.g. 9/12/38 of 59 matrices
    for a 70/15/15 target); see tests/test_splits_regression.py.

    Guarantee: when there are at least 3 groups, val and test are both non-empty
    (a split that stays empty receives the smallest remaining group).
    """
    from collections import defaultdict

    group_map: Dict[str, List[str]] = defaultdict(list)
    for mid in sorted(matrix_ids):
        group_map[groups.get(mid, "unknown")].append(mid)

    names = sorted(group_map.keys())
    rng.shuffle(names)                                   # random tie-breaking
    names.sort(key=lambda g: -len(group_map[g]))         # stable: largest first

    n_total = len(matrix_ids)
    targets = {
        "train": train_ratio * n_total,
        "val": val_ratio * n_total,
        "test": (1.0 - train_ratio - val_ratio) * n_total,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    manifest: SplitManifest = {}

    for idx, g in enumerate(names):
        size = len(group_map[g])
        remaining = len(names) - idx - 1
        empty = [s for s in ("test", "val") if counts[s] == 0 and targets[s] > 0]
        if empty and remaining < len(empty):
            # not enough groups left to fill every empty split: fill one now
            label = empty[0]
        else:
            def cost(s: str) -> float:
                return sum(
                    ((counts[t] + (size if t == s else 0)) - targets[t]) ** 2 for t in counts
                )
            label = min(("train", "val", "test"), key=lambda s: (cost(s), s))
        counts[label] += size
        for mid in group_map[g]:
            manifest[mid] = label

    for mid in matrix_ids:  # defensive; every id belongs to some group above
        manifest.setdefault(mid, "train")

    achieved = {k: v / n_total for k, v in counts.items()}
    want = {"train": train_ratio, "val": val_ratio, "test": 1.0 - train_ratio - val_ratio}
    worst = max(abs(achieved[k] - want[k]) for k in want)
    if worst > 0.10:
        logger.warning(
            "Group split deviates from target proportions by up to %.1f percentage points "
            "(achieved train/val/test = %.2f/%.2f/%.2f). Few or very unequal groups make "
            "exact proportions impossible; the achieved sizes are what is evaluated.",
            100 * worst, achieved["train"], achieved["val"], achieved["test"],
        )
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
