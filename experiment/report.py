"""
report.py
=========
Writes ``results/<run_id>/report.md`` and the figures from the recorded experiment tables.
Every number in the report is read from a table produced by ``experiment.run``; the narrative that
interprets them lives in ``docs/results.md`` and is written by a human after reading this report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from experiment import figures
from experiment.run import repeatability


def _fmt(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "n/a"
        if np.isinf(v):
            return "never"
        return f"{v:.4g}"
    return str(v)


def md_table(df: pd.DataFrame, cols: Optional[list] = None, pct: tuple = ()) -> str:
    d = df if cols is None else df[cols]
    d = d.rename(columns=str)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "|".join(["---"] * len(d.columns)) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            cells.append(f"{100 * v:.1f}%" if c in pct and isinstance(v, (float, np.floating)) and not np.isnan(v) else _fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(root: Path, res: Dict[str, Any]) -> Path:
    run_dir = Path(res["run_dir"])
    fig_dir = run_dir / "figures"
    rep = None
    rep_dir = run_dir.parent / f"{run_dir.name}_repeat" / "benchmark" / "benchmark_summary.csv"
    rep_tables: Optional[Dict[str, pd.DataFrame]] = None
    if rep_dir.is_file():
        rep_tables = repeatability(res["summary"], pd.read_csv(rep_dir))
        for k, v in rep_tables.items():
            v.to_csv(run_dir / f"exp6_repeatability_{k}.csv", index=False)
        rep = rep_tables["pairs"]
    figs = figures.make_all(res, res["timings_all"], res["feats_all"], fig_dir, rep)

    ov, per, paired, amort = res["overall"], res["per"], res["paired"], res["amortised"]
    ds, tol = res["ds"], res["tie"]
    n = {s: len(ds.matrix_ids(s)) for s in ("train", "val", "test")}
    groups = {s: ds.features_for(s)["matrix_group"].nunique() for s in ("train", "val", "test")}
    L = []
    L.append(f"# Experiment report: `{run_dir.name}`\n")
    L.append("> Auto-generated from recorded tables. Interpretation: `docs/results.md`. The oracle is an experimental upper bound, "
             "not a deployable method. Matrices of one group are not independent; with so few groups, intervals are wide and "
             "differences are rarely distinguishable from chance.\n")
    L.append("## 1. Data\n")
    L.append(f"* Matrices (after duplicate-pattern removal): train {n['train']} ({groups['train']} groups), validation {n['val']} "
             f"({groups['val']} groups), test {n['test']} ({groups['test']} groups); test matrices scored: {res['per']['matrix_id'].nunique()}.")
    L.append(f"* Kernels: {', '.join(ds.candidate_kernels)}. Best fixed kernel on TRAIN: `{res['fm'].notes.get('best_fixed_kernel')}`.")
    L.append(f"* Tie tolerance (measurement-based, train+val only; {tol['method']}): **{tol['tie_tolerance']:.0%}**. "
             f"Other noise statistics: median relative IQR of the winner {tol['median_relative_iqr_of_winner']:.2%}, "
             f"run-to-run median difference {tol.get('median_run_to_run_rel_diff_of_winner', float('nan')):.2%} "
             f"(90th percentile {tol.get('p90_run_to_run_rel_diff_of_winner', float('nan')):.2%}).")
    L.append(f"* Row-length heuristic learned on train: `{res['fm'].notes.get('rowlen_rule')}`.\n")
    L.append("## 2. Experiment 1: kernel characterisation (all matrices, descriptive)\n")
    L.append(md_table(res["char"]["kernel_summary"]) + "\n")
    L.append("Winning kernel by group (count of matrices):\n")
    L.append(md_table(res["char"]["wins_by_group"]) + "\n")
    L.append("## 3. Experiments 2-3: selection on the held-out test split\n")
    cols = ["method", "n_matrices", "n_groups", "accuracy", "mean_loss", "median_loss", "p90_loss", "max_loss", "group_mean_loss",
            "mean_loss_ci_low", "mean_loss_ci_high", "within_5pct", "within_10pct", "within_25pct", "total_time_ratio",
            "n_incorrect_selection", "selection_overhead_ms_median"]
    L.append(md_table(ov, cols, pct=("accuracy", "mean_loss", "median_loss", "p90_loss", "max_loss", "group_mean_loss",
                                     "mean_loss_ci_low", "mean_loss_ci_high", "within_5pct", "within_10pct", "within_25pct")) + "\n")
    L.append("Loss = selected kernel time / oracle kernel time - 1 (kernel time only). CI: 95% bootstrap over GROUPS.\n")
    L.append("### Paired comparisons of `rf_original` (loss difference, negative = selector better; group-level sign-flip test, Holm-corrected)\n")
    L.append(md_table(paired, ["method_b", "n_groups", "n_matrices", "mean_group_diff", "effect_size_dz", "ci_low", "ci_high",
                               "p_value", "p_floor", "p_holm"]) + "\n")
    L.append("`p_floor` is the smallest p-value attainable with that many groups; nothing smaller may be claimed.\n")
    L.append("## 4. Experiment 4: feature ablation (grouped CV on train+val; test not used)\n")
    L.append(md_table(res["ablation"].drop(columns=["features"]), pct=("cv_mean_loss", "cv_group_mean_loss", "cv_ci_low", "cv_ci_high", "cv_accuracy")) + "\n")
    L.append("## 5. Experiment 5: overhead and amortisation (test split)\n")
    L.append(md_table(amort[amort["method"].isin(["rf_original", "decision_tree", "rowlen_heuristic", "fixed_best_train", "fixed_cusparse"])],
                      ["method", "n_iterations", "total_ms", "oracle_total_ms", "ratio_to_oracle", "n_incorrect_selection"]) + "\n")
    L.append("Break-even number of SpMV products (selector must beat the fixed kernel including its own overhead):\n")
    L.append(md_table(res["break_even"][res["break_even"]["versus"] == "fixed_best_train"]) + "\n")
    L.append("## 6. Experiment 6: repeatability\n")
    if rep_tables is not None:
        pm, pr = rep_tables["per_matrix"], rep_tables["pairs"]
        L.append(f"* {len(pm)} matrices benchmarked twice (independent processes, same machine).")
        L.append(f"* Median |relative difference| of kernel medians: {pr['rel_diff'].abs().median():.2%}; 90th percentile {pr['rel_diff'].abs().quantile(0.9):.2%}.")
        L.append(f"* Same winning kernel in both runs: {pm['winner_agrees'].mean():.1%} of matrices.")
        gap = pm["runner_up_gap_run1"]
        L.append(f"* Matrices where the runner-up is within the tie tolerance ({tol['tie_tolerance']:.0%}): {(gap <= tol['tie_tolerance']).mean():.1%}.\n")
    else:
        L.append("Repeat run not available for this run id.\n")
    L.append("## 7. Sensitivity\n")
    L.append("Accuracy of the frozen selections under different tie tolerances (scoring only):\n")
    L.append(md_table(res["sensitivity"].pivot(index="method", columns="tie_tolerance", values="accuracy").reset_index(),) + "\n")
    L.append("## 8. Figures\n")
    for k, p in figs.items():
        L.append(f"![{k}](figures/{Path(p).name})")
    L.append("")
    out = run_dir / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out
