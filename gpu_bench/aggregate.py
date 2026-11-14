"""
aggregate.py
============
Turn the raw per-repetition output of ``spmv_bench`` into the project's CSV artifacts.

Nothing here is edited by hand: ``kernel_timings.csv`` is a pure function of the raw
measurements.  Statistics policy:

* ``kernel_time_ms``        = median of the timed repetitions (robust to OS jitter)
* ``preprocessing_time_ms`` = median of the repeated preprocessing measurements
* outliers (Tukey 1.5 * IQR fences) are COUNTED and reported but never removed
* a kernel that failed at run time gets NO row in kernel_timings.csv (no zeros, no
  imputation); it is written to benchmark_failures.csv with the reason
* ``correctness_pass`` is True only if the kernel ran to completion AND passed the
  tolerance check both on its first (cold, NaN-poisoned output) run and after the
  timed repetitions
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TIMING_COLUMNS = ["matrix_id", "kernel_name", "kernel_time_ms", "preprocessing_time_ms", "correctness_pass"]
SUMMARY_COLUMNS = [
    "matrix_id", "kernel_name", "n_reps", "median_ms", "mean_ms", "std_ms", "iqr_ms", "min_ms",
    "max_ms", "cv", "n_outliers", "prep_median_ms", "prep_std_ms", "timing_mode", "inner_iters",
    "max_abs_err", "threshold", "err_over_threshold", "correct_first", "correct_last",
]
FAILURE_COLUMNS = ["matrix_id", "kernel_name", "stage", "reason"]


def describe_samples(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("no samples")
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {
        "n": int(x.size), "median": float(med), "mean": mean, "std": std, "iqr": float(iqr),
        "min": float(x.min()), "max": float(x.max()), "cv": std / mean if mean > 0 else float("nan"),
        "n_outliers": int(((x < lo) | (x > hi)).sum()),
    }


def aggregate_matrix(matrix_id: str, part_dir: Path) -> Tuple[List[dict], List[dict], List[dict], Dict[str, str]]:
    """Return (timing_rows, summary_rows, failure_rows, aux dict) for one matrix."""
    kern = pd.read_csv(part_dir / f"{matrix_id}.kernels.csv")
    raw = pd.read_csv(part_dir / f"{matrix_id}.raw.csv")
    aux_df = pd.read_csv(part_dir / f"{matrix_id}.aux.csv", dtype=str)
    aux = dict(zip(aux_df["key"], aux_df["value"]))

    timing_rows, summary_rows, failure_rows = [], [], []
    for _, k in kern.iterrows():
        name = k["kernel"]
        if k["status"] != "ok":
            failure_rows.append({"matrix_id": matrix_id, "kernel_name": name, "stage": "run",
                                 "reason": str(k["error"])})
            continue
        t = raw[(raw["phase"] == "kernel") & (raw["kernel"] == name)]["ms"].to_numpy()
        p = raw[(raw["phase"] == "prep") & (raw["kernel"] == name)]["ms"].to_numpy()
        if t.size == 0 or p.size == 0:
            failure_rows.append({"matrix_id": matrix_id, "kernel_name": name, "stage": "aggregate",
                                 "reason": "missing raw samples"})
            continue
        ts, ps = describe_samples(t), describe_samples(p)
        passed = bool(int(k["correct_first"])) and bool(int(k["correct_last"]))
        thr = float(k["threshold"])
        timing_rows.append({"matrix_id": matrix_id, "kernel_name": name, "kernel_time_ms": ts["median"],
                            "preprocessing_time_ms": ps["median"], "correctness_pass": passed})
        summary_rows.append({
            "matrix_id": matrix_id, "kernel_name": name, "n_reps": ts["n"], "median_ms": ts["median"],
            "mean_ms": ts["mean"], "std_ms": ts["std"], "iqr_ms": ts["iqr"], "min_ms": ts["min"],
            "max_ms": ts["max"], "cv": ts["cv"], "n_outliers": ts["n_outliers"],
            "prep_median_ms": ps["median"], "prep_std_ms": ps["std"], "timing_mode": k["timing_mode"],
            "inner_iters": int(k["inner_iters"]),
            "max_abs_err": float(k["max_abs_err"]), "threshold": thr,
            "err_over_threshold": float(k["max_abs_err"]) / thr if thr > 0 else float("nan"),
            "correct_first": bool(int(k["correct_first"])), "correct_last": bool(int(k["correct_last"])),
        })
    return timing_rows, summary_rows, failure_rows, aux


def aux_row(matrix_id: str, aux: Dict[str, str]) -> dict:
    """Per-matrix overhead measurements (transfers, feature gathering, timer floor)."""
    def f(key: str) -> float:
        try:
            return float(aux[key])
        except (KeyError, ValueError):
            return float("nan")
    return {
        "matrix_id": matrix_id, "alloc_ms": f("alloc_ms"), "h2d_ms": f("h2d_ms_median"),
        "feature_cpu_ms": f("feature_cpu_ms_median"), "feature_gpu_ms": f("feature_gpu_ms_median"),
        "feature_gpu_matches_cpu": aux.get("feature_gpu_matches_cpu", ""),
        "timer_floor_ms": f("timer_floor_ms_median"),
        "mem_needed_mb_est": f("mem_needed_mb_est"),
    }


def write_frames(out_dir: Path, timing_rows, summary_rows, failure_rows, aux_rows, raw_frames) -> None:
    """Write the aggregate artifacts. Sorted for deterministic output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = pd.DataFrame(timing_rows, columns=TIMING_COLUMNS).sort_values(["matrix_id", "kernel_name"])
    timings.to_csv(out_dir / "kernel_timings.csv", index=False)
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).sort_values(["matrix_id", "kernel_name"]).to_csv(
        out_dir / "benchmark_summary.csv", index=False)
    pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS).to_csv(out_dir / "benchmark_failures.csv", index=False)
    pd.DataFrame(aux_rows).sort_values("matrix_id").to_csv(out_dir / "overhead_measurements.csv", index=False)
    if raw_frames:
        pd.concat(raw_frames).to_csv(out_dir / "raw_measurements.csv", index=False)
    else:
        pd.DataFrame(columns=["matrix_id", "phase", "kernel", "rep", "ms"]).to_csv(
            out_dir / "raw_measurements.csv", index=False)
