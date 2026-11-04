"""
train.py
========
Fit the selector on the train split, choose hyperparameters on the
validation split, and record everything needed to reproduce the run.

No-leakage contract
-------------------
This module only ever asks the dataset for the ``"train"`` and ``"val"``
splits. It never requests ``"test"`` (a test replaces the dataset accessors
with ones that raise on ``"test"`` to prove it).

Tuning objective: validation **oracle-relative loss**, not accuracy, because
accuracy ignores how costly a mistake is. Candidates are ranked by
(1) fewest picks of a kernel that failed correctness, (2) lowest mean loss,
(3) highest accuracy, (4) position in the grid (so ties are deterministic).

Open decision Q2: after tuning, refit on train only (default) or on
train+val (``refit_on_train_val=True``)? Declare the choice in the config
*before* looking at the test split.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics import score_selections
from selector.data import FEATURE_COLUMNS, Dataset
from selector.labels import DEFAULT_TIE_TOLERANCE, oracle_table
from selector.model import DEFAULT_PARAMS, KernelSelector, library_versions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_grid() -> Dict[str, List[Any]]:
    return {
        "n_estimators": [100, 300],
        "max_depth": [None, 4, 8],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }


@dataclass(frozen=True)
class TrainConfig:
    feature_columns: Sequence[str] = tuple(FEATURE_COLUMNS)
    seed: int = 0
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE
    grid: Mapping[str, Sequence[Any]] = field(default_factory=_default_grid)
    refit_on_train_val: bool = False


@dataclass
class TrainResult:
    selector: KernelSelector
    best_params: Dict[str, Any]
    tuning: pd.DataFrame
    train_no_oracle: pd.DataFrame
    card: Dict[str, Any]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def expand_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of the grid, in a deterministic (sorted-key) order."""
    unknown = set(grid) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown grid parameter(s): {sorted(unknown)}; allowed: {sorted(DEFAULT_PARAMS)}")
    keys = sorted(grid)
    if any(len(grid[k]) == 0 for k in keys):
        raise ValueError("every grid parameter needs at least one value")
    if not keys:
        return [{}]
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _labelled(dataset: Dataset, split: str, tol: float):
    """Feature rows of one split joined with oracle labels (matrices without an oracle dropped)."""
    oracle, no_oracle = oracle_table(dataset.timings_for(split), tol)
    labels = oracle[["matrix_id", "oracle_kernel"]].rename(columns={"oracle_kernel": "label"})
    data = dataset.features_for(split).merge(labels, on="matrix_id", how="inner", validate="one_to_one")
    return data, no_oracle


def _validation_score(selector: KernelSelector, dataset: Dataset, tol: float) -> Optional[Dict[str, float]]:
    """Score a fitted selector on the validation split. None if nothing is scorable."""
    val_timings = dataset.timings_for("val")
    oracle, _ = oracle_table(val_timings, tol)
    feats = dataset.features_for("val")
    feats = feats[feats["matrix_id"].isin(set(oracle["matrix_id"]))]
    if feats.empty:
        return None
    picks = pd.DataFrame({"matrix_id": feats["matrix_id"], "selected_kernel": selector.predict(feats)})
    scores = score_selections(picks, val_timings, oracle, tol)
    losses = scores["oracle_relative_loss"].to_numpy(dtype=float)
    return {
        "n_val": int(len(scores)),
        "n_incorrect_selection": int((~scores["selected_correct_flag"]).sum()),
        "mean_val_loss": float(np.nanmean(losses)) if np.isfinite(losses).any() else float("nan"),
        "val_accuracy": float(scores["selection_correct"].mean()),
    }


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

def train_selector(
    dataset: Dataset,
    config: TrainConfig = TrainConfig(),
    input_paths: Optional[Mapping[str, Path | str]] = None,
) -> TrainResult:
    """Train + tune. ``input_paths`` (name -> file) is only used for provenance hashes."""
    tol = config.tie_tolerance
    features = list(config.feature_columns)

    train_data, train_no_oracle = _labelled(dataset, "train", tol)
    if train_data.empty:
        raise ValueError("no train matrices with an oracle label; cannot train")

    combos = expand_grid(config.grid)
    rows = []
    for order, params in enumerate(combos):
        sel = KernelSelector(features, config.seed, **params)
        sel.fit(train_data[features], train_data["label"])
        score = _validation_score(sel, dataset, tol)
        if score is None:
            if len(combos) > 1:
                raise ValueError(
                    "no validation matrices with an oracle; cannot tune a grid of "
                    f"{len(combos)} settings. Provide validation data or a single-setting grid."
                )
            score = {"n_val": 0, "n_incorrect_selection": 0,
                     "mean_val_loss": float("nan"), "val_accuracy": float("nan")}
        rows.append({"grid_order": order, **{k: params.get(k) for k in sorted(config.grid)}, **score})

    tuning = pd.DataFrame(rows)
    rank_key = tuning.assign(
        _loss=tuning["mean_val_loss"].fillna(np.inf), _neg_acc=-tuning["val_accuracy"].fillna(-1.0)
    ).sort_values(["n_incorrect_selection", "_loss", "_neg_acc", "grid_order"])
    best_idx = int(rank_key.index[0])
    tuning["chosen"] = tuning.index == best_idx
    best_params = combos[best_idx]

    if config.refit_on_train_val:
        val_data, _ = _labelled(dataset, "val", tol)
        final_data = pd.concat([train_data, val_data], ignore_index=True)
    else:
        final_data = train_data
    final = KernelSelector(features, config.seed, **best_params)
    final.fit(final_data[features], final_data["label"])

    hashes = {name: file_sha256(p) for name, p in (input_paths or {}).items()}
    synthetic = any("synthetic" in Path(p).name.lower() for p in (input_paths or {}).values())
    cfg = asdict(config)
    cfg["feature_columns"] = list(cfg["feature_columns"])
    card = {
        "config": cfg,
        "best_params": {**DEFAULT_PARAMS, **best_params},
        "classes": final.classes_,
        "candidate_kernels": list(dataset.candidate_kernels),
        "feature_columns": features,
        "n_train_matrices_used": int(len(train_data)),
        "n_final_fit_matrices": int(len(final_data)),
        "n_train_without_oracle": int(len(train_no_oracle)),
        "refit_on_train_val": config.refit_on_train_val,
        "versions": library_versions(),
        "git_commit": _git_commit(),
        "input_sha256": hashes,
        "synthetic_data": synthetic,
    }
    return TrainResult(final, best_params, tuning, train_no_oracle, card)


def write_training_outputs(result: TrainResult, out_dir: Path | str) -> Dict[str, Path]:
    """Write model.joblib (gitignored), model_card.json and tuning_results.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": result.selector.save(out / "model.joblib"),
        "model_card": out / "model_card.json",
        "tuning": out / "tuning_results.csv",
    }
    card_text = json.dumps(result.card, indent=2, sort_keys=True, default=str)
    paths["model_card"].write_text(card_text, encoding="utf-8")
    result.tuning.to_csv(paths["tuning"], index=False)
    return paths
