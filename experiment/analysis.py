"""
analysis.py
===========
The frozen evaluation plan for the real experiment.

Order of operations (enforced by ``experiment.run``):

1.  ``measure_tie_tolerance``      from repetition noise of TRAIN+VAL matrices only
2.  ``fit_methods``                every method is fitted / tuned with TRAIN (fit) and VAL (tuning) only
3.  ``feature_ablation``           grouped CV on TRAIN+VAL only
4.  ``evaluate_methods('test')``   ONE scoring pass on the held-out test split of the already-frozen methods
5.  statistics / overhead / sensitivity on the per-matrix test table

No function in steps 1-3 ever requests the test split.  Step 4 is guarded by a one-shot flag file in
``experiment.run``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.group_stats import cluster_bootstrap_ci, group_mean, holm_adjust, paired_group_signflip_test
from evaluation.metrics import fraction_within, score_selections
from experiment import methods as M
from experiment import overhead as O
from selector.baselines import best_fixed_on_train, fixed_selections, majority_kernel_on_train
from selector.data import FEATURE_COLUMNS, Dataset
from selector.labels import oracle_table
from selector.model import KernelSelector
from selector.train import TrainConfig, train_selector

ORACLE = "oracle"
BASIC_SIZE = ["rows", "cols", "nnz", "mean_row_nnz"]
ROW_DIST = ["mean_row_nnz", "variance_row_nnz", "max_row_nnz", "min_row_nnz"]
EXTENDED = ["cv_row_nnz", "empty_row_frac", "max_over_mean", "frac_rows_gt_32"]
FEATURE_SETS: Dict[str, List[str]] = {
    "size_only": ["rows", "cols", "nnz"],
    "basic_size": BASIC_SIZE,
    "row_distribution": ROW_DIST,
    "original_7": list(FEATURE_COLUMNS),
    "extended_11": list(FEATURE_COLUMNS) + EXTENDED,
}
# Features that need a pass over the row pointer (cost counted); size features come free with the CSR header.
NEEDS_ROW_PASS = {"variance_row_nnz", "max_row_nnz", "min_row_nnz", "cv_row_nnz", "empty_row_frac",
                  "max_over_mean", "frac_rows_gt_32"}

RF_GRID = {"n_estimators": [100, 300], "max_depth": [None, 4, 8], "min_samples_leaf": [1, 2],
           "class_weight": [None, "balanced"]}
DT_GRID = {"max_depth": [2, 3, 4, 6, None], "min_samples_leaf": [1, 2, 4]}


# ------------------------------------------------------------------ #
# 1. measurement-based tie tolerance (train + val only)                #
# ------------------------------------------------------------------ #

def measure_tie_tolerance(summary: pd.DataFrame, train_val_ids: Sequence[str], floor: float = 0.01,
                          summary_repeat: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Measurement-based tie tolerance, from TRAIN+VAL matrices only.

    Two kernels closer than the measurement noise cannot be reliably ranked.  The noise that matters for
    ranking kernels by their MEDIAN time is the run-to-run difference of medians, so when an independent
    repeat benchmark is available the tolerance is the median over train+val matrices of
    |median_run2 - median_run1| / median_run1 for the fastest kernel, rounded up to a whole percent (never below
    ``floor``).  Without a repeat run the (much larger, single-repetition) relative IQR of the winner is used
    as a fallback and the method is recorded.  Both statistics are always reported.
    """
    ids = set(train_val_ids)
    s = summary[summary["matrix_id"].isin(ids) & summary["correct_first"] & summary["correct_last"]]
    best = s.sort_values("median_ms").groupby("matrix_id").head(1)
    rel_iqr = (best["iqr_ms"] / best["median_ms"]).to_numpy(float)
    out: Dict[str, float] = {
        "median_relative_iqr_of_winner": float(np.median(rel_iqr)) if rel_iqr.size else float("nan"),
        "p90_relative_iqr_of_winner": float(np.percentile(rel_iqr, 90)) if rel_iqr.size else float("nan"),
        "n_matrices": int(rel_iqr.size),
    }
    if summary_repeat is not None:
        r = summary_repeat.set_index(["matrix_id", "kernel_name"])["median_ms"]
        rel = np.array([abs(r.get((m, k), np.nan) - v) / v for m, k, v in zip(best["matrix_id"], best["kernel_name"], best["median_ms"])])
        rel = rel[~np.isnan(rel)]
        out["median_run_to_run_rel_diff_of_winner"] = float(np.median(rel))
        out["p90_run_to_run_rel_diff_of_winner"] = float(np.percentile(rel, 90))
        stat, method = out["median_run_to_run_rel_diff_of_winner"], "median run-to-run difference of medians (train+val)"
    else:
        stat, method = out["median_relative_iqr_of_winner"], "median relative IQR of the winner (train+val), fallback: no repeat run"
    out["method"] = method
    out["tie_tolerance"] = max(floor, float(np.ceil(stat * 100) / 100)) if np.isfinite(stat) else floor
    return out


# ------------------------------------------------------------------ #
# 2. fit methods                                                       #
# ------------------------------------------------------------------ #

@dataclass
class FittedMethods:
    selectors: Dict[str, Any] = field(default_factory=dict)      # name -> object with predict(X)
    needs_rows: Dict[str, bool] = field(default_factory=dict)    # does the method need the row-pass features?
    tuning: Dict[str, pd.DataFrame] = field(default_factory=dict)
    fixed: Dict[str, str] = field(default_factory=dict)          # name -> kernel
    notes: Dict[str, Any] = field(default_factory=dict)


def _with_extended(feats: pd.DataFrame, ext: Optional[pd.DataFrame]) -> pd.DataFrame:
    if ext is None:
        return feats
    return feats.merge(ext, on="matrix_id", how="left", validate="one_to_one")


def fit_methods(dataset: Dataset, ext: Optional[pd.DataFrame], tie_tol: float, seed: int = 0,
                extended_feature_set: Optional[str] = None) -> FittedMethods:
    """Fit/tune all methods on TRAIN and VAL only."""
    out = FittedMethods()
    tr_f = _with_extended(dataset.features_for("train"), ext)
    va_f = _with_extended(dataset.features_for("val"), ext)
    tr_t, va_t = dataset.timings_for("train"), dataset.timings_for("val")

    # (a) existing Random Forest (Member 3), unchanged pipeline, 7 features
    rf = train_selector(dataset, TrainConfig(seed=seed, tie_tolerance=tie_tol))
    out.selectors["rf_original"] = rf.selector
    out.tuning["rf_original"] = rf.tuning
    out.needs_rows["rf_original"] = True
    out.notes["rf_original_params"] = rf.best_params

    # (b) decision tree, original 7 features
    dt, dt_tab = M.tune_on_validation(lambda **p: M.TreeSelector(FEATURE_SETS["original_7"], seed, **p), DT_GRID,
                                      tr_f, tr_t, va_f, va_t, tie_tol)
    out.selectors["decision_tree"], out.tuning["decision_tree"], out.needs_rows["decision_tree"] = dt, dt_tab, True

    # (c) row-length heuristic fitted on train timings
    h = M.RowLengthHeuristic().fit(tr_f, tr_t)
    out.selectors["rowlen_heuristic"], out.needs_rows["rowlen_heuristic"] = h, False  # mean_row_nnz = nnz/rows
    out.notes["rowlen_rule"] = h.describe()

    # (d) optional RF on the feature set that grouped CV (train+val) preferred
    if extended_feature_set and extended_feature_set != "original_7":
        cols = FEATURE_SETS[extended_feature_set]
        rfx, tab = M.tune_on_validation(lambda **p: KernelSelector(cols, seed, **p), RF_GRID, tr_f, tr_t, va_f, va_t, tie_tol)
        name = f"rf_{extended_feature_set}"
        out.selectors[name], out.tuning[name] = rfx, tab
        out.needs_rows[name] = any(c in NEEDS_ROW_PASS for c in cols)

    # (e) fixed baselines (train-only choice for the "best fixed")
    best_fixed, table = best_fixed_on_train(dataset)
    out.fixed["fixed_best_train"] = best_fixed
    out.fixed["majority_label_train"] = majority_kernel_on_train(dataset, tie_tol)
    for k in dataset.candidate_kernels:
        out.fixed[f"fixed_{k}"] = k
    out.notes["best_fixed_kernel"] = best_fixed
    out.notes["train_kernel_table"] = table.to_dict("records")
    return out


# ------------------------------------------------------------------ #
# 3. feature ablation (train + val only, grouped CV)                    #
# ------------------------------------------------------------------ #

def feature_ablation(dataset: Dataset, ext: pd.DataFrame, tie_tol: float, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Grouped K-fold CV over TRAIN+VAL matrices for every feature set (default RF settings).

    Reports mean loss, group-weighted mean loss, accuracy and a cluster-bootstrap CI.  The test split
    is not touched.  Also reports the trivial majority baseline for context."""
    feats = pd.concat([dataset.features_for("train"), dataset.features_for("val")], ignore_index=True)
    feats = _with_extended(feats, ext)
    timings = pd.concat([dataset.timings_for("train"), dataset.timings_for("val")], ignore_index=True)
    rows = []
    for name, cols in FEATURE_SETS.items():
        res = M.grouped_cv_loss(lambda: KernelSelector(cols, seed), cols, feats, timings, k, seed, tie_tol)
        loss = res["oracle_relative_loss"].to_numpy(float)
        lo, hi = cluster_bootstrap_ci(loss, res["matrix_group"], seed=seed)
        rows.append({"feature_set": name, "n_features": len(cols), "features": ";".join(cols), "n_matrices": int(len(res)),
                     "n_groups": int(res["matrix_group"].nunique()),
                     "cv_mean_loss": float(np.nanmean(loss)), "cv_group_mean_loss": group_mean(loss, res["matrix_group"]),
                     "cv_ci_low": lo, "cv_ci_high": hi, "cv_accuracy": float(res["selection_correct"].mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# 4. evaluation on one split                                           #
# ------------------------------------------------------------------ #

def inference_ms(selector, row: pd.DataFrame, repeats: int = 5) -> float:
    """Median wall-clock ms of a single-matrix predict() call on this machine's CPU."""
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        selector.predict(row)
        vals.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(vals))


def method_selections(fm: FittedMethods, feats: pd.DataFrame, oracle: pd.DataFrame,
                      feature_cpu_ms: Mapping[str, float]) -> Dict[str, Dict[str, Any]]:
    """Selections + per-matrix selection overhead (ms) for every method on ``feats``."""
    ids = feats["matrix_id"].tolist()
    out: Dict[str, Dict[str, Any]] = {}
    for name, sel in fm.selectors.items():
        picks, over = [], {}
        for i in range(len(feats)):
            row = feats.iloc[[i]]
            mid = row["matrix_id"].iloc[0]
            picks.append(str(sel.predict(row)[0]))
            over[mid] = inference_ms(sel, row) + (float(feature_cpu_ms.get(mid, 0.0)) if fm.needs_rows[name] else 0.0)
        out[name] = {"selections": pd.DataFrame({"matrix_id": ids, "selected_kernel": picks}), "overhead_ms": over}
    for name, kernel in fm.fixed.items():
        out[name] = {"selections": fixed_selections(ids, kernel), "overhead_ms": {}}
    out[ORACLE] = {"selections": oracle[oracle["matrix_id"].isin(ids)][["matrix_id", "oracle_kernel"]]
                   .rename(columns={"oracle_kernel": "selected_kernel"}), "overhead_ms": {}}
    return out


def score_all(sel: Mapping[str, Dict[str, Any]], timings: pd.DataFrame, oracle: pd.DataFrame,
              groups: Mapping[str, str], tol: float) -> pd.DataFrame:
    parts = []
    for name, d in sel.items():
        sc = score_selections(d["selections"], timings, oracle, tol)
        sc.insert(0, "method", name)
        sc["selection_overhead_ms"] = sc["matrix_id"].map(d["overhead_ms"]).fillna(0.0)
        sc["matrix_group"] = sc["matrix_id"].map(groups)
        parts.append(sc)
    return pd.concat(parts, ignore_index=True)


def summarize(per: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per method: matrix-level and group-aware statistics (kernel time only)."""
    rows = []
    for name, d in per.groupby("method", sort=False):
        loss = d["oracle_relative_loss"].to_numpy(float)
        ok = d["selected_correct_flag"].to_numpy(bool)
        lo, hi = cluster_bootstrap_ci(loss, d["matrix_group"], n_boot=n_boot, seed=seed)
        fw = fraction_within(loss)
        rows.append({
            "method": name, "n_matrices": int(len(d)), "n_groups": int(d["matrix_group"].nunique()),
            "accuracy": float(d["selection_correct"].mean()),
            "mean_loss": float(np.nanmean(loss)) if np.isfinite(loss).any() else float("nan"),
            "median_loss": float(np.nanmedian(loss)) if np.isfinite(loss).any() else float("nan"),
            "p90_loss": float(np.nanpercentile(loss, 90)) if np.isfinite(loss).any() else float("nan"),
            "max_loss": float(np.nanmax(loss)) if np.isfinite(loss).any() else float("nan"),
            "group_mean_loss": group_mean(loss, d["matrix_group"]),
            "mean_loss_ci_low": lo, "mean_loss_ci_high": hi,
            "within_5pct": fw[0.05], "within_10pct": fw[0.10], "within_25pct": fw[0.25],
            "total_time_ratio": float(d.loc[ok, "selected_kernel_time_ms"].sum() / d.loc[ok, "oracle_time_ms"].sum()) if ok.any() else float("nan"),
            "n_incorrect_selection": int((~ok).sum()),
            "selection_overhead_ms_median": float(d["selection_overhead_ms"].median()),
        })
    return pd.DataFrame(rows)


def paired_comparisons(per: pd.DataFrame, a: str, others: Sequence[str], seed: int = 0) -> pd.DataFrame:
    """Method ``a`` versus each of ``others`` on the same matrices: difference in oracle-relative
    loss (a - other; negative = a better), tested at GROUP level with Holm correction across the family."""
    base = per[per["method"] == a].set_index("matrix_id")
    rows = []
    for o in others:
        cmp = per[per["method"] == o].set_index("matrix_id")
        ids = sorted(set(base.index) & set(cmp.index))
        diff = base.loc[ids, "oracle_relative_loss"].to_numpy(float) - cmp.loc[ids, "oracle_relative_loss"].to_numpy(float)
        r = paired_group_signflip_test(diff, base.loc[ids, "matrix_group"].to_numpy(), seed=seed)
        rows.append({"method_a": a, "method_b": o, **r})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm_adjust(df["p_value"].to_numpy(float))
    return df


def confusion(per: pd.DataFrame, method: str, kernels: Sequence[str]) -> pd.DataFrame:
    d = per[per["method"] == method]
    return pd.crosstab(pd.Categorical(d["oracle_kernel"], categories=list(kernels)),
                       pd.Categorical(d["selected_kernel"], categories=list(kernels)), dropna=False
                       ).rename_axis(index="oracle", columns="selected")


def tolerance_sensitivity(per_sel: pd.DataFrame, timings: pd.DataFrame, tolerances: Sequence[float]) -> pd.DataFrame:
    """Accuracy of already-made selections under different tie tolerances (scoring only; nothing is refit)."""
    rows = []
    for tol in tolerances:
        oracle, _ = oracle_table(timings, tol)
        for name, d in per_sel.groupby("method", sort=False):
            sc = score_selections(d[["matrix_id"]].assign(selected_kernel=d["selected_kernel"].to_numpy()), timings, oracle, tol)
            rows.append({"tie_tolerance": tol, "method": name, "accuracy": float(sc["selection_correct"].mean())})
    return pd.DataFrame(rows)
