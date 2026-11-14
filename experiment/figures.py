"""
figures.py
==========
Publication figures generated ONLY from the recorded experiment tables.

Design rules (dataviz skill): categorical colours in fixed order from a palette that passes the CVD /
lightness checks, never cycled; a legend whenever there are >= 2 series; distinct marker shapes as secondary
encoding; thin marks; recessive grid; bars anchored at zero; no dual axes; nothing omitted (every matrix is
plotted, under-performing ones included).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KERNEL_COLOR = {"csr_scalar": "#2a78d6", "csr_vector": "#eb6834", "csr_adaptive": "#1baf7a", "cusparse": "#eda100"}
KERNEL_MARK = {"csr_scalar": "o", "csr_vector": "s", "csr_adaptive": "^", "cusparse": "D"}
METHOD_COLOR = {"rf_original": "#2a78d6", "decision_tree": "#6250d6", "rowlen_heuristic": "#1baf7a",
                "fixed_best_train": "#eb6834", "oracle": "#6b6b66"}
INK, MUTED, GRID = "#1a1a19", "#6b6b66", "#e3e2dc"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 100})


def _save(fig, path: Path, note: Optional[str] = None) -> Path:
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _kc(k: str) -> str:
    return KERNEL_COLOR.get(k, MUTED)


def fig_kernel_comparison(timings: pd.DataFrame, path: Path) -> Path:
    """Fig 1: per-matrix slowdown of every kernel relative to the fastest correct kernel (all matrices)."""
    ok = timings[timings["correctness_pass"].astype(bool)]
    w = ok.pivot(index="matrix_id", columns="kernel_name", values="kernel_time_ms").dropna()
    slow = w.div(w.min(axis=1), axis=0)
    kernels = list(slow.columns)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    data = [slow[k].to_numpy() for k in kernels]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=True, patch_artist=True,
                    flierprops={"marker": ".", "markersize": 3, "markerfacecolor": MUTED, "markeredgecolor": "none"},
                    medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, k in zip(bp["boxes"], kernels):
        patch.set_facecolor(_kc(k)); patch.set_alpha(0.85); patch.set_edgecolor(MUTED)
    ax.set_yticks(range(1, len(kernels) + 1)); ax.set_yticklabels(kernels)
    ax.set_xscale("log"); ax.axvline(1.0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Kernel time / fastest kernel's time on the same matrix (x, log scale; 1 = best)")
    ax.set_title(f"Kernel performance comparison across {len(w)} real matrices (box = IQR, line = median)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def _scatter_by_kernel(df: pd.DataFrame, xcol: str, ycol: str, path: Path, xlabel: str, ylabel: str, title: str,
                       logy: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k in sorted(df["kernel_name"].unique()):
        d = df[df["kernel_name"] == k]
        ax.scatter(d[xcol], d[ycol], s=16, marker=KERNEL_MARK.get(k, "o"), color=_kc(k), alpha=0.75,
                   edgecolors="white", linewidths=0.4, label=k)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, title="kernel")
    return _save(fig, path)


def fig_time_vs_nnz(timings: pd.DataFrame, feats: pd.DataFrame, path: Path) -> Path:
    d = timings[timings["correctness_pass"].astype(bool)].merge(feats[["matrix_id", "nnz"]], on="matrix_id")
    d["time_us"] = d["kernel_time_ms"] * 1e3
    return _scatter_by_kernel(d, "nnz", "time_us", path, "Number of nonzeros (log scale)", "Kernel time per SpMV (µs, log scale)",
                              "Kernel time versus matrix size (every matrix, every kernel)")


def fig_slowdown_vs_variance(timings: pd.DataFrame, feats: pd.DataFrame, path: Path) -> Path:
    ok = timings[timings["correctness_pass"].astype(bool)]
    w = ok.pivot(index="matrix_id", columns="kernel_name", values="kernel_time_ms").dropna()
    slow = w.div(w.min(axis=1), axis=0).stack().rename("slowdown").reset_index()
    d = slow.merge(feats[["matrix_id", "variance_row_nnz"]], on="matrix_id")
    d["variance_row_nnz"] = d["variance_row_nnz"] + 1e-3   # zero variance is possible; keeps the log axis defined
    return _scatter_by_kernel(d, "variance_row_nnz", "slowdown", path, "Row-length variance (population, log scale; +0.001 offset)",
                              "Slowdown vs fastest kernel (x, log scale)", "Relative kernel performance versus row-length irregularity")


def fig_selector_vs_baselines(overall: pd.DataFrame, path: Path) -> Path:
    d = overall.copy()
    d = d[d["method"] != "oracle"].sort_values("mean_loss")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(d))
    cols = [METHOD_COLOR.get(m, "#9a9a94") for m in d["method"]]
    ax.barh(y, d["mean_loss"] * 100, color=cols, height=0.6)
    lo = np.nan_to_num(d["mean_loss"].to_numpy() - d["mean_loss_ci_low"].to_numpy())
    hi = np.nan_to_num(d["mean_loss_ci_high"].to_numpy() - d["mean_loss"].to_numpy())
    ax.errorbar(d["mean_loss"] * 100, y, xerr=[lo * 100, hi * 100], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(d["method"]); ax.invert_yaxis()
    ax.set_xlabel("Mean loss vs oracle (%; kernel time only; bars from 0)")
    ax.set_title("Selector versus baselines on the held-out test split\n(error bars: 95% group-level bootstrap CI; oracle = 0 by definition)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path, "Oracle is an experimental upper bound, not a deployable method.")


def fig_loss_distribution(per: pd.DataFrame, path: Path, methods=("rf_original", "decision_tree", "rowlen_heuristic", "fixed_best_train")) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m in methods:
        d = per[per["method"] == m]["oracle_relative_loss"].dropna().sort_values().to_numpy() * 100
        if d.size == 0:
            continue
        ax.step(d, np.arange(1, d.size + 1) / d.size * 100, where="post",
                color=METHOD_COLOR.get(m, MUTED), linewidth=1.8, label=m)
    ax.set_xlabel("Oracle-relative loss on a matrix (%)"); ax.set_ylabel("Fraction of test matrices with loss ≤ x (%)")
    ax.set_title("Distribution of oracle-relative loss (test split, every matrix included)")
    ax.legend(frameon=False, loc="lower right"); ax.set_ylim(0, 101)
    return _save(fig, path)


def fig_per_group(per: pd.DataFrame, path: Path, methods=("rf_original", "fixed_best_train", "rowlen_heuristic", "decision_tree")) -> Path:
    d = per[per["method"].isin(methods)].groupby(["matrix_group", "method"])["oracle_relative_loss"].mean().unstack() * 100
    d = d.reindex(columns=[m for m in methods if m in d.columns])
    n = per[per["method"] == methods[0]].groupby("matrix_group").size()
    d = d.loc[d.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(d) + 1.8))
    y = np.arange(len(d))
    for m, mark in zip(d.columns, ["o", "s", "^", "D"]):
        ax.scatter(d[m], y, marker=mark, s=36, color=METHOD_COLOR.get(m, MUTED), label=m, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([f"{g} (n={n[g]})" for g in d.index]); ax.invert_yaxis()
    ax.set_xlabel("Mean oracle-relative loss in the group (%)"); ax.set_title("Per-group performance on the test split")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path, "Groups with one or two matrices are anecdotal; do not read them as group-level conclusions.")


def fig_confusion(conf: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(conf.to_numpy(), cmap="Blues", vmin=0)
    ax.set_xticks(range(conf.shape[1])); ax.set_xticklabels(conf.columns, rotation=30, ha="right")
    ax.set_yticks(range(conf.shape[0])); ax.set_yticklabels(conf.index)
    thr = conf.to_numpy().max() / 2 if conf.to_numpy().size else 0
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            v = int(conf.iat[i, j]); ax.text(j, i, v, ha="center", va="center", color="white" if v > thr else INK)
    ax.set_xlabel("Selected kernel (rf_original)"); ax.set_ylabel("Oracle (fastest) kernel"); ax.set_title("Confusion matrix, test split (counts)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="matrices")
    return _save(fig, path)


def fig_ablation(ab: pd.DataFrame, path: Path) -> Path:
    d = ab.sort_values("cv_group_mean_loss")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(d))
    ax.barh(y, d["cv_group_mean_loss"] * 100, color="#2a78d6", height=0.55)
    lo = (d["cv_group_mean_loss"] - d["cv_ci_low"]).clip(lower=0).fillna(0).to_numpy() * 100
    hi = (d["cv_ci_high"] - d["cv_group_mean_loss"]).clip(lower=0).fillna(0).to_numpy() * 100
    ax.errorbar(d["cv_group_mean_loss"] * 100, y, xerr=[lo, hi], fmt="none", ecolor=INK, capsize=3, elinewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{f} ({n})" for f, n in zip(d["feature_set"], d["n_features"])]); ax.invert_yaxis()
    ax.set_xlabel("Grouped-CV group-mean loss vs oracle (%; train+val matrices only; bars from 0)")
    ax.set_title("Feature ablation (leave-groups-out cross-validation; number of features in brackets)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path, "Test split not used. Error bars: 95% cluster-bootstrap CI of the mean over matrices.")


def fig_break_even(amort: pd.DataFrame, path: Path, methods=("rf_original", "decision_tree", "rowlen_heuristic", "fixed_best_train")) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m in methods:
        d = amort[amort["method"] == m].sort_values("n_iterations")
        if d.empty:
            continue
        ax.plot(d["n_iterations"], d["ratio_to_oracle"], marker="o", markersize=4, linewidth=1.8, color=METHOD_COLOR.get(m, MUTED), label=m)
    ax.axhline(1.0, color=METHOD_COLOR["oracle"], linewidth=1, linestyle="--", label="oracle (no overhead, N-aware)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SpMV products per matrix, N (log scale)"); ax.set_ylabel("Total time / N-aware oracle total time (x, log scale)")
    ax.set_title("End-to-end cost including selection overhead and preprocessing (test split)")
    ax.legend(frameon=False)
    return _save(fig, path, "Selection overhead = feature gathering + selector inference (wall clock, CPU). Preprocessing charged once per matrix.")


def fig_repeatability(rep: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for k in sorted(rep["kernel_name"].unique()):
        d = rep[rep["kernel_name"] == k]
        ax.scatter(d["median_ms_run1"] * 1e3, d["median_ms_run2"] * 1e3, s=14, marker=KERNEL_MARK.get(k, "o"), color=_kc(k),
                   alpha=0.75, edgecolors="white", linewidths=0.3, label=k)
    lim = [rep[["median_ms_run1", "median_ms_run2"]].min().min() * 1e3 * 0.9, rep[["median_ms_run1", "median_ms_run2"]].max().max() * 1e3 * 1.1]
    ax.plot(lim, lim, color=MUTED, linewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Median kernel time, run 1 (µs, log)"); ax.set_ylabel("Median kernel time, repeat run (µs, log)")
    ax.set_title("Repeatability: two independent benchmark runs (line = identical)")
    ax.legend(frameon=False, title="kernel")
    return _save(fig, path)


def make_all(res: Mapping[str, object], timings_all: pd.DataFrame, feats: pd.DataFrame, fig_dir: Path,
             repeat: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
    out = {
        "fig01_kernel_comparison": fig_kernel_comparison(timings_all, fig_dir / "fig01_kernel_comparison.png"),
        "fig02_time_vs_nnz": fig_time_vs_nnz(timings_all, feats, fig_dir / "fig02_time_vs_nnz.png"),
        "fig03_slowdown_vs_variance": fig_slowdown_vs_variance(timings_all, feats, fig_dir / "fig03_slowdown_vs_variance.png"),
        "fig04_selector_vs_baselines": fig_selector_vs_baselines(res["overall"], fig_dir / "fig04_selector_vs_baselines.png"),
        "fig05_loss_distribution": fig_loss_distribution(res["per"], fig_dir / "fig05_loss_distribution.png"),
        "fig06_per_group": fig_per_group(res["per"], fig_dir / "fig06_per_group.png"),
        "fig07_confusion": fig_confusion(res["confusion"], fig_dir / "fig07_confusion.png"),
        "fig08_feature_ablation": fig_ablation(res["ablation"], fig_dir / "fig08_feature_ablation.png"),
        "fig09_break_even": fig_break_even(res["amortised"], fig_dir / "fig09_break_even.png"),
    }
    if repeat is not None and len(repeat):
        out["fig10_repeatability"] = fig_repeatability(repeat, fig_dir / "fig10_repeatability.png")
    return out
