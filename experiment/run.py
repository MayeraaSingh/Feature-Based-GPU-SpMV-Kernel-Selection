"""
run.py
======
End-to-end experiment driver:  benchmark artifacts  ->  frozen methods  ->  one test pass  ->  tables.

Hard refusals (clear error, non-zero exit, nothing written for the test split):
  * required input files are missing or violate the CSV contracts (``selector.data`` validation)
  * any input name / manifest row is marked synthetic
  * the benchmark is incomplete: a candidate kernel is missing for more than ``max_incomplete_fraction``
    of the matrices, or a candidate kernel never passed correctness
  * the test split of this run directory has already been evaluated (one-shot ``TEST_EVALUATED.json``)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from experiment import analysis as A
from experiment import overhead as O
from selector.data import DataContractError, build_dataset

N_VALUES = (1, 10, 100, 1000, 10000, 100000)
TOLERANCES = (0.0, 0.01, 0.02, 0.05, 0.10)


class RefusedToRun(RuntimeError):
    """Raised for conditions under which the experiment must not proceed."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def preflight(root: Path, cfg: Dict[str, Any], bench_dir: Path, max_incomplete_fraction: float = 0.2) -> Dict[str, Path]:
    """Check everything that must hold before any model is fitted. Returns the resolved input paths."""
    meta = root / cfg["dataset"]["metadata_dir"]
    paths = {"features": meta / "features.csv", "splits": meta / "splits.csv", "catalog": meta / "matrix_catalog.csv",
             "timings": bench_dir / "kernel_timings.csv", "summary": bench_dir / "benchmark_summary.csv",
             "overhead": bench_dir / "overhead_measurements.csv", "failures": bench_dir / "benchmark_failures.csv",
             "extended": meta / "features_extended.csv", "manifest": root / cfg["dataset"].get("manifest", "")}
    for key in ("features", "splits", "catalog", "timings", "summary", "overhead", "failures", "extended", "manifest"):
        if not paths[key].is_file():
            raise RefusedToRun(f"required input missing: {paths[key]} ({key})")
    for p in paths.values():
        if "synthetic" in p.name.lower():
            raise RefusedToRun(f"synthetic input refused for a real experiment: {p}")
    man = pd.read_csv(paths["manifest"])
    if man["source"].astype(str).str.contains("synthetic", case=False).any():
        raise RefusedToRun("the dataset manifest contains synthetic sources")
    env = bench_dir / "environment.json"
    if not env.is_file():
        raise RefusedToRun(f"benchmark environment record missing: {env}")
    try:
        ds = build_dataset(paths["features"], paths["splits"], paths["catalog"], paths["timings"])
    except DataContractError as exc:
        raise RefusedToRun(f"CSV contract violation: {exc}") from exc
    n_all = len(pd.read_csv(paths["features"]))
    if len(ds.excluded) > max_incomplete_fraction * n_all:
        raise RefusedToRun(f"timing data incomplete: {len(ds.excluded)} of {n_all} matrices lack a candidate kernel "
                           f"(limit {max_incomplete_fraction:.0%}); see {paths['failures']}")
    t = pd.read_csv(paths["timings"])
    for k in ds.candidate_kernels:
        if not t.loc[t["kernel_name"] == k, "correctness_pass"].astype(bool).any():
            raise RefusedToRun(f"candidate kernel '{k}' never passed correctness; fix the kernel before the experiment")
    if len(ds.candidate_kernels) < 2:
        raise RefusedToRun("fewer than two candidate kernels")
    return paths


def _kernel_characterisation(timings: pd.DataFrame, feats: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Experiment 1: descriptive kernel comparison over the whole benchmark (no model involved)."""
    ok = timings[timings["correctness_pass"].astype(bool)]
    wide = ok.pivot(index="matrix_id", columns="kernel_name", values="kernel_time_ms").dropna()
    best = wide.idxmin(axis=1)
    f = feats.set_index("matrix_id").loc[wide.index]
    summ = pd.DataFrame({
        "kernel": wide.columns,
        "wins": [int((best == k).sum()) for k in wide.columns],
        "median_ms": [float(wide[k].median()) for k in wide.columns],
        "geomean_slowdown_vs_best": [float(np.exp(np.log(wide[k] / wide.min(axis=1)).mean())) for k in wide.columns],
    })
    if "cusparse" in wide.columns:
        summ["geomean_speedup_vs_cusparse"] = [float(np.exp(np.log(wide["cusparse"] / wide[k]).mean())) for k in wide.columns]
    by_group = pd.crosstab(f["matrix_group"], best).reindex(columns=wide.columns, fill_value=0)
    corr_rows = []
    for k in wide.columns:
        rel = np.log(wide[k] / wide.min(axis=1))
        for feat in ("nnz", "rows", "mean_row_nnz", "variance_row_nnz", "max_row_nnz"):
            corr_rows.append({"kernel": k, "feature": feat, "spearman_with_log_slowdown": float(rel.corr(f[feat], method="spearman"))})
    return {"kernel_summary": summ, "wins_by_group": by_group.reset_index(), "feature_correlation": pd.DataFrame(corr_rows)}


def run_experiment(root: Path, run_dir: Path, config_path: Path, bench_dir: Optional[Path] = None, seed: int = 0,
                   n_boot: int = 2000) -> Dict[str, Any]:
    root, run_dir = Path(root), Path(run_dir)
    bench_dir = Path(bench_dir) if bench_dir else run_dir / "benchmark"
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    flag = run_dir / "TEST_EVALUATED.json"
    if flag.exists():
        raise RefusedToRun(f"the test split of run '{run_dir.name}' was already evaluated ({flag}); "
                           "use a new run directory. Re-evaluating the test set after seeing results is not allowed.")
    paths = preflight(root, cfg, bench_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    ds = build_dataset(paths["features"], paths["splits"], paths["catalog"], paths["timings"])
    ext = pd.read_csv(paths["extended"])
    summary = pd.read_csv(paths["summary"])
    overhead = pd.read_csv(paths["overhead"]).set_index("matrix_id")
    feature_cpu = overhead["feature_cpu_ms"].to_dict()
    groups = dict(zip(ds.features["matrix_id"], ds.features["matrix_group"]))
    split_ids = {s: ds.matrix_ids(s) for s in ("train", "val", "test")}

    # ---- steps 1-3: only train + val information ----------------------------------------------
    rep_path = run_dir.parent / f"{run_dir.name}_repeat" / "benchmark" / "benchmark_summary.csv"
    rep_summary = pd.read_csv(rep_path) if rep_path.is_file() else None
    tol = A.measure_tie_tolerance(summary, split_ids["train"] + split_ids["val"], summary_repeat=rep_summary)
    ablation = A.feature_ablation(ds, ext, tol["tie_tolerance"], seed=seed)
    ablation.to_csv(run_dir / "feature_ablation_cv.csv", index=False)
    best_row = ablation.sort_values(["cv_group_mean_loss", "n_features"]).iloc[0]
    fm = A.fit_methods(ds, ext, tol["tie_tolerance"], seed=seed, extended_feature_set=str(best_row["feature_set"]))
    tuning = pd.concat([t.assign(method=n) for n, t in fm.tuning.items()], ignore_index=True)
    tuning.to_csv(run_dir / "tuning_results.csv", index=False)

    # ---- step 4: ONE pass over the test split -------------------------------------------------
    flag.write_text(json.dumps({"evaluated_at": pd.Timestamp.now("UTC").isoformat(),
                                "git_commit": _git(root, "rev-parse", "HEAD")}, indent=2), encoding="utf-8")
    test_t = ds.timings_for("test")
    test_f = A._with_extended(ds.features_for("test"), ext)
    oracle, no_oracle = A.oracle_table(test_t, tol["tie_tolerance"])
    test_f = test_f[test_f["matrix_id"].isin(set(oracle["matrix_id"]))].reset_index(drop=True)
    sel = A.method_selections(fm, test_f, oracle, feature_cpu)
    per = A.score_all(sel, test_t, oracle, groups, tol["tie_tolerance"])
    per.to_csv(run_dir / "per_matrix_test.csv", index=False)
    overall = A.summarize(per, n_boot, seed)
    overall.to_csv(run_dir / "summary_overall.csv", index=False)
    by_group = (per.groupby(["method", "matrix_group"]).agg(n=("matrix_id", "size"), mean_loss=("oracle_relative_loss", "mean"),
                accuracy=("selection_correct", "mean")).reset_index())
    by_group.to_csv(run_dir / "summary_by_group.csv", index=False)
    kernels = list(ds.candidate_kernels)
    conf = A.confusion(per, "rf_original", kernels)
    conf.to_csv(run_dir / "confusion_matrix.csv")

    # ---- statistics / overhead / sensitivity (test table only) --------------------------------
    fixed_names = [n for n in fm.fixed]
    comps = [m for m in ["fixed_best_train", "rowlen_heuristic", "decision_tree", "majority_label_train"] +
             [f"fixed_{k}" for k in kernels] if m in sel]
    comps = list(dict.fromkeys(comps))
    paired = A.paired_comparisons(per, "rf_original", comps, seed=seed)
    paired.to_csv(run_dir / "paired_comparisons.csv", index=False)
    extra_methods = [m for m in fm.selectors if m not in ("rf_original", "decision_tree", "rowlen_heuristic")]
    if extra_methods:
        pd.concat([A.paired_comparisons(per, m, ["rf_original", "fixed_best_train"], seed=seed) for m in extra_methods]
                  ).to_csv(run_dir / "paired_comparisons_extra.csv", index=False)

    amort_methods = {m: d["selections"] for m, d in sel.items() if m != A.ORACLE}
    amort_over = {m: d["overhead_ms"] for m, d in sel.items()}
    amort = O.amortised_table(amort_methods, test_t, amort_over, N_VALUES)
    amort.to_csv(run_dir / "amortised_cost.csv", index=False)
    be_rows = []
    for m in [x for x in fm.selectors]:
        for other in ["fixed_best_train"] + [f"fixed_{k}" for k in kernels]:
            n_star = O.break_even(sel[m]["selections"], sel[other]["selections"], test_t, sel[m]["overhead_ms"], sel[other]["overhead_ms"])
            be_rows.append({"selector": m, "versus": other, "break_even_iterations": n_star})
    pd.DataFrame(be_rows).to_csv(run_dir / "break_even.csv", index=False)

    sens_methods = ["rf_original", "decision_tree", "rowlen_heuristic", "fixed_best_train", ""]
    sens = A.tolerance_sensitivity(per[per["method"].isin([m for m in sens_methods if m])], test_t, TOLERANCES)
    sens.to_csv(run_dir / "tolerance_sensitivity.csv", index=False)
    big = set(ds.features.loc[ds.features["nnz"] >= 100_000, "matrix_id"])
    A.summarize(per[per["matrix_id"].isin(big)], n_boot, seed).assign(subset="nnz>=100k").to_csv(
        run_dir / "summary_large_matrices.csv", index=False)

    # Experiment 1 on the whole benchmark (descriptive)
    char = _kernel_characterisation(pd.read_csv(paths["timings"]), ds.features)
    for k, v in char.items():
        v.to_csv(run_dir / f"exp1_{k}.csv", index=False)

    # ---- provenance -------------------------------------------------------------------------
    shutil.copy(config_path, run_dir / "config.yaml")
    for name in ("environment.json",):
        shutil.copy(bench_dir / name, run_dir / name)
    summary.to_csv(run_dir / "benchmark_summary.csv", index=False)
    rf_model = fm.selectors["rf_original"]
    rf_model.save(run_dir / "model.joblib")
    manifest = {
        "run_id": run_dir.name, "git": {"commit": _git(root, "rev-parse", "HEAD"), "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
                                        "dirty": bool(_git(root, "status", "--porcelain"))},
        "seeds": {"split_seed": cfg["splits"]["random_seed"], "model_seed": seed, "bootstrap": seed},
        "tie_tolerance": tol, "n_matrices": {s: len(v) for s, v in split_ids.items()},
        "n_test_scored": int(len(test_f)), "test_without_oracle": no_oracle.to_dict("records"),
        "excluded_incomplete": ds.excluded.to_dict("records"),
        "candidate_kernels": kernels, "ablation_selected_feature_set": str(best_row["feature_set"]),
        "rowlen_rule": fm.notes.get("rowlen_rule"), "best_fixed_on_train": fm.notes.get("best_fixed_kernel"),
        "input_sha256": {k: _sha256(p) for k, p in paths.items()},
        "benchmark_manifest": json.loads((bench_dir / "benchmark_manifest.json").read_text()) if (bench_dir / "benchmark_manifest.json").is_file() else None,
        "timing_protocol": "see docs/benchmark_protocol.md",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (run_dir / "dataset_manifest.json").write_text(json.dumps({
        "source": "NIST Matrix Market", "manifest_csv_sha256": manifest["input_sha256"]["manifest"],
        "n_matrices": int(len(pd.read_csv(paths["features"]))), "groups": sorted(set(groups.values())),
        "split_counts": manifest["n_matrices"], "split_seed": cfg["splits"]["random_seed"]}, indent=2), encoding="utf-8")
    (run_dir / "model_card.json").write_text(json.dumps({
        "rf_original": {"params": fm.notes.get("rf_original_params"), "features": rf_model.feature_columns, "classes": rf_model.classes_},
        "decision_tree_params": fm.selectors["decision_tree"].params, "rowlen_rule": fm.notes.get("rowlen_rule"),
        "feature_importance_rf_original": dict(zip(rf_model.feature_columns, map(float, rf_model._model.feature_importances_))),
        "versions": rf_model.versions}, indent=2, default=str), encoding="utf-8")

    return {"run_dir": run_dir, "per": per, "overall": overall, "paired": paired, "amortised": amort, "ablation": ablation,
            "tie": tol, "fm": fm, "char": char, "sel": sel, "ds": ds, "test_timings": test_t, "break_even": pd.DataFrame(be_rows),
            "sensitivity": sens, "summary": summary, "overhead": overhead, "confusion": conf,
            "timings_all": pd.read_csv(paths["timings"]), "feats_all": ds.features}


def repeatability(summary1: pd.DataFrame, summary2: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Experiment 6: compare two independent benchmark runs of the same matrices (no model involved).

    Per (matrix, kernel): relative difference of medians.  Per matrix: does the winner agree, and is the
    winner distinguishable (runner-up more than the run-to-run noise slower)?"""
    a = summary1.set_index(["matrix_id", "kernel_name"])["median_ms"].rename("median_ms_run1")
    b = summary2.set_index(["matrix_id", "kernel_name"])["median_ms"].rename("median_ms_run2")
    d = pd.concat([a, b], axis=1).dropna().reset_index()
    d["rel_diff"] = (d["median_ms_run2"] - d["median_ms_run1"]) / d["median_ms_run1"]
    rows = []
    for mid, g in d.groupby("matrix_id"):
        w1, w2 = g.loc[g["median_ms_run1"].idxmin(), "kernel_name"], g.loc[g["median_ms_run2"].idxmin(), "kernel_name"]
        s1 = np.sort(g["median_ms_run1"].to_numpy())
        rows.append({"matrix_id": mid, "winner_run1": w1, "winner_run2": w2, "winner_agrees": w1 == w2,
                     "runner_up_gap_run1": float(s1[1] / s1[0] - 1) if len(s1) > 1 else float("nan")})
    return {"pairs": d, "per_matrix": pd.DataFrame(rows)}
