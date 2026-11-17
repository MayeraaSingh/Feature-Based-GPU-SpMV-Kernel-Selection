"""
run_gpu_benchmark.py
====================
Benchmark every matrix of the real dataset with the CUDA kernels and write the benchmark
artifacts (kernel_timings.csv, raw_measurements.csv, benchmark_summary.csv,
overhead_measurements.csv, host_stage_timings.csv, gpu_telemetry.csv, benchmark_failures.csv,
environment.json, benchmark_manifest.json) into --out-dir.

    python scripts/run_gpu_benchmark.py --out-dir results/<run_id>/benchmark
    python scripts/run_gpu_benchmark.py --out-dir ... --matrix-ids a,b,c      # subset (repeatability)

Refuses to run when the executable or CUDA GPU is missing, or matrix files are missing.
Never writes into benchmarks/kernel_timings.csv unless that path is given explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
import yaml

from gpu_bench import runner
from gpu_bench.environment import capture_environment, write_environment


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_root / "configs/dataset_real.yaml"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--timing", choices=["graph", "direct"], default="graph")
    ap.add_argument("--matrix-ids", default="", help="comma-separated subset; default: all in features.csv")
    ap.add_argument("--exe", default=None)
    args = ap.parse_args(argv)

    if args.warmup < 10 or args.reps < 50:
        print("ERROR: protocol minimum is warmup >= 10 and reps >= 50.", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = _root / cfg["dataset"]["raw_dir"]
    meta = _root / cfg["dataset"]["metadata_dir"]
    try:
        exe = runner.exe_path(Path(args.exe) if args.exe else None)
    except runner.BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    feats = meta / "features.csv"
    if not feats.is_file():
        print(f"ERROR: {feats} missing; run scripts/run_pipeline.py --config {args.config}", file=sys.stderr)
        return 2
    ids = [i for i in args.matrix_ids.split(",") if i] or sorted(pd.read_csv(feats)["matrix_id"])
    missing = [i for i in ids if not (raw_dir / f"{i}.mtx").is_file()]
    if missing:
        print(f"ERROR: {len(missing)} matrix file(s) missing in {raw_dir}: {missing[:5]}...", file=sys.stderr)
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = capture_environment(_root, exe)
    write_environment(out / "environment.json", env)
    res = runner.benchmark_matrices(raw_dir, ids, out, exe=exe, warmup=args.warmup, reps=args.reps,
                                    eps=args.eps, timing=args.timing)
    manifest = {
        "kernels": list(runner.KERNELS), "n_matrices_requested": len(ids),
        "warmup": args.warmup, "reps": args.reps, "epsilon": args.eps, "timing_mode": args.timing,
        "precision": "float32 GPU compute, float64 CPU reference",
        "input_vector": "uniform[-1,1) float32, seed 42",
        "adaptive_thresholds": "row length <=4:1 thread, <=16:4, <=64:16, <=512:warp, >512:block(256); fixed, not tuned",
        "kernel_timings_rows": int(len(res["timings"])), "failures": int(len(res["failures"])),
        "git": env["git"],
    }
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"timings: {len(res['timings'])} rows, failures: {len(res['failures'])} -> {out}")
    return 0 if res["failures"].empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
