"""
tables.py
=========
Summary tables (CSV + Markdown) built from the per-matrix result table.

Loss columns are FRACTIONS: 0.10 means "10 % slower than the oracle".
Accuracy is always reported next to loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.evaluate import SELECTOR, fixed_method, method_order
from evaluation.metrics import fraction_within, summarize_scores, total_time_ratio

SYNTHETIC_BANNER = "SYNTHETIC DATA - NOT A RESULT"
REAL_PROVENANCE = "measured"
THRESHOLDS = (0.0, 0.05, 0.10, 0.25)


def provenance(synthetic: bool) -> str:
    return SYNTHETIC_BANNER if synthetic else REAL_PROVENANCE


def _summarize_method(rows: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, float]:
    s = summarize_scores(rows, n_boot=n_boot, seed=seed)
    # summarize_scores' "n" is the number of matrices with a defined loss.
    s["n_with_loss"] = s.pop("n")
    for t, v in fraction_within(rows["oracle_relative_loss"], THRESHOLDS).items():
        s[f"within_{int(round(t * 100))}pct"] = v
    ok = rows["selected_correct_flag"]
    s["total_end_to_end_ratio"] = total_time_ratio(
        rows.loc[ok, "selected_end_to_end_ms"], rows.loc[ok, "oracle_end_to_end_ms"]
    )
    s["mean_selection_overhead_ms"] = float(rows["selection_overhead_ms"].mean()) if rows[
        "selection_overhead_ms"].notna().any() else float("nan")
    return s


def overall_table(
    per_matrix: pd.DataFrame, best_fixed_kernel: Optional[str], n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """One row per method, ordered selector / best fixed / other fixed / oracle."""
    best = fixed_method(best_fixed_kernel) if best_fixed_kernel else None
    rows = []
    for method in method_order(per_matrix, best_fixed_kernel):
        s = _summarize_method(per_matrix[per_matrix["method"] == method], n_boot, seed)
        rows.append({"method": method, "is_best_fixed": method == best, **s})
    return pd.DataFrame(rows)


def group_table(
    per_matrix: pd.DataFrame, best_fixed_kernel: Optional[str], n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Selector vs best-fixed baseline per ``matrix_group`` (shows behaviour on unseen families)."""
    methods = [SELECTOR] + ([fixed_method(best_fixed_kernel)] if best_fixed_kernel else [])
    rows = []
    for group, sub in per_matrix[per_matrix["method"].isin(methods)].groupby("matrix_group", sort=True):
        for method in methods:
            part = sub[sub["method"] == method]
            if part.empty:
                continue
            s = summarize_scores(part, n_boot=n_boot, seed=seed)
            rows.append({
                "matrix_group": group, "method": method, "n_matrices": s["n_matrices"],
                "accuracy": s["accuracy"], "n_incorrect_selection": s["n_incorrect_selection"],
                "n_with_loss": s["n"], "mean_loss": s["mean_loss"], "median_loss": s["median_loss"],
                "mean_ci_low": s["mean_ci_low"], "mean_ci_high": s["mean_ci_high"],
            })
    cols = ["matrix_group", "method", "n_matrices", "accuracy", "n_incorrect_selection",
            "n_with_loss", "mean_loss", "median_loss", "mean_ci_low", "mean_ci_high"]
    return pd.DataFrame(rows, columns=cols)


def confusion_table(per_matrix: pd.DataFrame, kernels: Sequence[str]) -> pd.DataFrame:
    """Counts of (oracle kernel -> selected kernel) for the selector. Every candidate appears."""
    sel = per_matrix[per_matrix["method"] == SELECTOR]
    table = pd.crosstab(sel["oracle_kernel"], sel["selected_kernel"])
    table = table.reindex(index=list(kernels), columns=list(kernels), fill_value=0)
    table.index.name = "oracle_kernel \\ selected_kernel"
    return table.astype(int)


def to_markdown(df: pd.DataFrame, banner: Optional[str] = None, note: Optional[str] = None,
                index: bool = False) -> str:
    """Small dependency-free GitHub-markdown table (no `tabulate` needed)."""
    frame = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.4f}"
        return str(v)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    parts = []
    if banner:
        parts.append(f"> **{banner}**\n")
    if note:
        parts.append(f"{note}\n")
    parts.append("\n".join([header, rule] + body) + "\n")
    return "\n".join(parts)


def write_report_tables(
    out_dir: Path | str,
    per_matrix: pd.DataFrame,
    excluded: pd.DataFrame,
    best_fixed_kernel: Optional[str],
    candidate_kernels: Sequence[str],
    synthetic: bool,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Path]:
    """Write all CSV/Markdown tables; synthetic runs are stamped everywhere."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance(synthetic)
    banner = SYNTHETIC_BANNER if synthetic else None
    note = "Loss columns are fractions (0.10 = 10 % slower than the oracle). The oracle is an upper bound."

    overall = overall_table(per_matrix, best_fixed_kernel, n_boot, seed)
    groups = group_table(per_matrix, best_fixed_kernel, n_boot, seed)
    confusion = confusion_table(per_matrix, candidate_kernels)

    paths: Dict[str, Path] = {}
    for name, df in (("per_matrix_test", per_matrix), ("summary_overall", overall),
                     ("summary_by_group", groups), ("excluded_matrices", excluded)):
        paths[name] = out / f"{name}.csv"
        df.assign(provenance=prov).to_csv(paths[name], index=False)
    paths["confusion_matrix"] = out / "confusion_matrix.csv"
    confusion.to_csv(paths["confusion_matrix"])
    for name, df, idx in (("summary_overall", overall, False), ("summary_by_group", groups, False),
                          ("confusion_matrix", confusion, True)):
        paths[name + "_md"] = out / f"{name}.md"
        paths[name + "_md"].write_text(to_markdown(df, banner, note, index=idx), encoding="utf-8")
    if synthetic:
        marker = out / "RESULTS_ARE_SYNTHETIC.txt"
        marker.write_text(
            "Every file in this folder was produced from SYNTHETIC timings.\n"
            "They exercise the code only and are NOT GPU measurements or scientific results.\n",
            encoding="utf-8")
        paths["synthetic_marker"] = marker
    return paths


def write_manifest(path: Path | str, manifest: Dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
