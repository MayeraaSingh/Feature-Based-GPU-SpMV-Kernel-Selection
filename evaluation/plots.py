"""
plots.py
========
Static PNG figures (matplotlib, headless 'Agg' backend).

Design (dataviz method, light surface)
--------------------------------------
* Colour = role, assigned in fixed order and validated with the dataviz
  palette validator (slots 1-2 pass adjacent + all-pairs CVD / normal-vision
  gates): selector = blue ``#2a78d6``, best fixed baseline = orange ``#eb6834``.
  The oracle is neutral ink (it is a reference bound, not a competitor); other
  fixed kernels are muted gray. Colour is never the only cue: every bar/line is
  also labelled on the axis or directly.
* Magnitude heatmap (confusion matrix) uses the one-hue blue sequential ramp.
* Thin marks, recessive horizontal grid, no top/right spines, text in ink
  tokens (never in the series colour).
* Synthetic runs carry a banner and a diagonal watermark on every figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless: works on servers, CI and Colab without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from evaluation.evaluate import ORACLE, SELECTOR, fixed_method, method_order  # noqa: E402
from evaluation.metrics import summarize_loss  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, CRITICAL = "#2a78d6", "#eb6834", "#d03b3b"
# One-hue sequential ramp, blue 100 -> 700 (light = few, dark = many).
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

BANNER = "SYNTHETIC - NOT A RESULT"
DPI = 150


def _color(method: str, best_fixed_method: Optional[str]) -> str:
    if method == SELECTOR:
        return BLUE
    if method == best_fixed_method:
        return ORANGE
    if method == ORACLE:
        return INK2
    return MUTED


def _label(method: str, best_fixed_method: Optional[str]) -> str:
    if method == SELECTOR:
        return "Selector"
    if method == ORACLE:
        return "Oracle (upper bound)"
    name = method.replace("fixed_", "always ")
    return f"Best fixed ({name})" if method == best_fixed_method else name


def _fig(width: float = 8.0, height: float = 4.8):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    return fig, ax


def _title(ax, title: str, subtitle: Optional[str] = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=22 if subtitle else 10, fontweight="bold")
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


def _save(fig, path: Path | str, synthetic: bool) -> Path:
    if synthetic:
        fig.text(0.99, 0.99, BANNER, ha="right", va="top", fontsize=9, color=CRITICAL, fontweight="bold")
        fig.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", fontsize=60, color=CRITICAL,
                 alpha=0.10, rotation=30, fontweight="bold")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _empty(path: Path | str, title: str, synthetic: bool) -> Path:
    fig, ax = _fig()
    ax.axis("off")
    ax.text(0.5, 0.5, "No matrices to plot", ha="center", va="center", color=MUTED, fontsize=12)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    return _save(fig, path, synthetic)


def _loss_pct(df: pd.DataFrame) -> np.ndarray:
    return df["oracle_relative_loss"].to_numpy(dtype=float) * 100.0


# ------------------------------------------------------------------ #
# Figures                                                              #
# ------------------------------------------------------------------ #

def plot_per_matrix_loss(per_matrix, best_fixed_kernel, path, synthetic=False, max_labels=40):
    """1. Per-matrix slowdown vs oracle, sorted by the selector's loss."""
    sel = per_matrix[per_matrix["method"] == SELECTOR].sort_values(
        "oracle_relative_loss", na_position="last")
    if sel.empty:
        return _empty(path, "Per-matrix loss vs oracle", synthetic)
    ids = sel["matrix_id"].tolist()
    best = fixed_method(best_fixed_kernel) if best_fixed_kernel else None
    series = [(SELECTOR, BLUE)] + ([(best, ORANGE)] if best else [])
    fig, ax = _fig(max(8.0, 0.35 * len(ids) + 3), 5.0)
    width = 0.38 if len(series) == 2 else 0.6
    x = np.arange(len(ids))
    for k, (method, color) in enumerate(series):
        sub = per_matrix[per_matrix["method"] == method].set_index("matrix_id").reindex(ids)
        loss = sub["oracle_relative_loss"].to_numpy(dtype=float) * 100.0
        offs = (k - (len(series) - 1) / 2) * width
        ax.bar(x + offs, np.nan_to_num(loss), width=width * 0.92, color=color,
               label=_label(method, best), zorder=3)
        if len(ids) <= 20:   # direct value labels; a 0 % bar would otherwise be invisible
            for xi, val in zip(x + offs, loss):
                if not np.isnan(val):
                    ax.text(xi, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=INK2)
        bad = np.isnan(loss)
        ax.scatter((x + offs)[bad], np.zeros(bad.sum()), marker="X", s=40, color=CRITICAL, zorder=4,
                   label="wrong result (no loss defined)" if bad.any() and k == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(ids if len(ids) <= max_labels else [""] * len(ids), rotation=60, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylabel("Slower than oracle")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _title(ax, "Per-matrix loss vs oracle", f"{len(ids)} test matrices, sorted by selector loss (0 % = optimal)")
    return _save(fig, path, synthetic)


def plot_loss_ecdf(per_matrix, best_fixed_kernel, path, synthetic=False):
    """2. Fraction of matrices within x % of the oracle. Wrong-result picks never count as 'within'."""
    best = fixed_method(best_fixed_kernel) if best_fixed_kernel else None
    methods = [SELECTOR] + ([best] if best else [])
    fig, ax = _fig()
    if per_matrix[per_matrix["method"] == SELECTOR].empty:
        plt.close(fig)
        return _empty(path, "Loss distribution", synthetic)
    finite = per_matrix[per_matrix["method"].isin(methods)]["oracle_relative_loss"].dropna()
    xmax = max(10.0, float(finite.max()) * 100.0 * 1.05) if len(finite) else 10.0
    for method in methods:
        loss = np.sort(np.nan_to_num(
            per_matrix[per_matrix["method"] == method]["oracle_relative_loss"].to_numpy(dtype=float),
            nan=np.inf)) * 100.0
        n = len(loss)
        xs = np.concatenate([[0.0], loss[np.isfinite(loss)], [xmax]])
        ys = np.concatenate([[0.0], np.arange(1, np.isfinite(loss).sum() + 1) / n,
                             [np.isfinite(loss).sum() / n]])
        color = _color(method, best)
        ax.step(xs, ys, where="post", color=color, linewidth=2, label=_label(method, best))
        # Direct label on the middle of the curve's longest flat stretch (keeps labels apart).
        k = int(np.argmax(np.diff(xs)))
        ax.annotate(_label(method, best), ((xs[k] + xs[k + 1]) / 2, ys[k]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Slower than oracle by at most ...")
    ax.set_ylabel("Share of test matrices")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    _title(ax, "How close to the oracle?", "Curves further up and to the left are better")
    return _save(fig, path, synthetic)


def plot_group_loss(per_matrix, best_fixed_kernel, path, synthetic=False, n_boot=2000, seed=0):
    """3. Mean loss per matrix group with bootstrap CI; n annotated."""
    best = fixed_method(best_fixed_kernel) if best_fixed_kernel else None
    methods = [SELECTOR] + ([best] if best else [])
    sub = per_matrix[per_matrix["method"].isin(methods)]
    groups = sorted(sub["matrix_group"].dropna().unique())
    if not groups:
        return _empty(path, "Loss by matrix group", synthetic)
    fig, ax = _fig(8.0, max(3.2, 0.9 * len(groups) + 1.8))
    height = 0.36 if len(methods) == 2 else 0.6
    y = np.arange(len(groups))
    for k, method in enumerate(methods):
        means, lo, hi = [], [], []
        for g in groups:
            s = summarize_loss(sub[(sub["method"] == method) & (sub["matrix_group"] == g)]
                               ["oracle_relative_loss"], n_boot=n_boot, seed=seed)
            means.append(s["mean_loss"] * 100.0)
            lo.append(s["mean_ci_low"] * 100.0)
            hi.append(s["mean_ci_high"] * 100.0)
        means, lo, hi = map(np.array, (means, lo, hi))
        offs = (k - (len(methods) - 1) / 2) * height
        ax.barh(y + offs, np.nan_to_num(means), height=height * 0.9, color=_color(method, best),
                label=_label(method, best), zorder=3)
        for yi, val in zip(y + offs, means):   # direct labels (a 0 % bar has no visible length)
            ax.text(0 if np.isnan(val) else val, yi, "  n/a" if np.isnan(val) else f"  {val:.1f}%",
                    va="center", ha="left", fontsize=8, color=INK2, zorder=5)
        has_ci = ~np.isnan(lo)
        ax.errorbar(means[has_ci], (y + offs)[has_ci],
                    xerr=[means[has_ci] - lo[has_ci], hi[has_ci] - means[has_ci]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    counts = [int((sub[(sub["method"] == SELECTOR) & (sub["matrix_group"] == g)]).shape[0]) for g in groups]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  (n={n})" for g, n in zip(groups, counts)], fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Mean slower-than-oracle (bars) with 95 % bootstrap CI (whiskers)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _title(ax, "Loss by matrix group", "Whiskers are absent where a group has fewer than 2 matrices")
    return _save(fig, path, synthetic)


def plot_confusion(confusion: pd.DataFrame, path, synthetic=False):
    """4. Oracle kernel (rows) vs selected kernel (columns), counts."""
    fig, ax = _fig(max(5.0, 1.0 * len(confusion.columns) + 3), max(4.2, 0.9 * len(confusion) + 2))
    data = confusion.to_numpy(dtype=float)
    ax.imshow(data, cmap=SEQ_CMAP, vmin=0, vmax=max(1.0, data.max()))
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(confusion.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(confusion.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dark_cell = data[i, j] > 0.55 * max(1.0, data.max())
            ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=11,
                    color="#ffffff" if dark_cell else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Selected kernel")
    ax.set_ylabel("Oracle (fastest correct) kernel")
    _title(ax, "Selector choice vs oracle", "Diagonal = agreed with the oracle; counts of test matrices")
    return _save(fig, path, synthetic)


def plot_total_runtime(per_matrix, best_fixed_kernel, path, synthetic=False):
    """5. Total kernel time per method over matrices where every method gave a correct result."""
    best = fixed_method(best_fixed_kernel) if best_fixed_kernel else None
    ok = per_matrix.groupby("matrix_id")["selected_correct_flag"].all()
    common = set(ok[ok].index)
    sub = per_matrix[per_matrix["matrix_id"].isin(common)]
    if sub.empty:
        return _empty(path, "Total kernel time", synthetic)
    methods = method_order(sub, best_fixed_kernel)
    totals = [float(sub.loc[sub["method"] == m, "selected_kernel_time_ms"].sum()) for m in methods]
    fig, ax = _fig(max(7.0, 1.1 * len(methods) + 2), 4.8)
    x = np.arange(len(methods))
    ax.bar(x, totals, width=0.6, color=[_color(m, best) for m in methods], zorder=3)
    for xi, t in zip(x, totals):
        ax.text(xi, t, f"{t:.3g}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m, best).replace(" (", "\n(") for m in methods], fontsize=8.5, color=INK2)
    ax.set_ylabel("Total kernel time, ms (lower is better)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    excluded = per_matrix["matrix_id"].nunique() - len(common)
    _title(ax, "Total kernel time over the test set",
           f"{len(common)} matrices where every method returned a correct result"
           + (f" ({excluded} left out)" if excluded else ""))
    return _save(fig, path, synthetic)


def plot_feature_importance(importances: pd.Series, path, synthetic=False):
    """6. Impurity-based importances (diagnostic only; biased toward high-variance features)."""
    imp = importances.sort_values()
    fig, ax = _fig(7.0, max(3.2, 0.45 * len(imp) + 1.8))
    ax.barh(range(len(imp)), imp.to_numpy(), color=BLUE, height=0.6, zorder=3)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.index, fontsize=9, color=INK2)
    ax.set_xlabel("Importance (sums to 1)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    _title(ax, "Feature importances", "Diagnostic only; not evidence of causal influence")
    return _save(fig, path, synthetic)


def make_all_plots(
    out_dir: Path | str,
    per_matrix: pd.DataFrame,
    confusion: pd.DataFrame,
    importances: pd.Series,
    best_fixed_kernel: Optional[str],
    synthetic: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Path]:
    out = Path(out_dir)
    return {
        "per_matrix_loss": plot_per_matrix_loss(per_matrix, best_fixed_kernel, out / "per_matrix_loss.png", synthetic),
        "loss_ecdf": plot_loss_ecdf(per_matrix, best_fixed_kernel, out / "loss_ecdf.png", synthetic),
        "group_loss": plot_group_loss(per_matrix, best_fixed_kernel, out / "group_loss.png", synthetic, n_boot, seed),
        "confusion": plot_confusion(confusion, out / "confusion_matrix.png", synthetic),
        "total_runtime": plot_total_runtime(per_matrix, best_fixed_kernel, out / "total_runtime.png", synthetic),
        "feature_importance": plot_feature_importance(importances, out / "feature_importance.png", synthetic),
    }
