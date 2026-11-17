"""
runner.py
=========
Run the CUDA benchmark executable on a set of validated matrices and aggregate results.

Flow per matrix: load + validate (timed, host wall clock) -> export GPU input -> run
``spmv_bench`` in its own process -> telemetry before/after -> aggregate.  Everything
produced here comes from the executable's raw output; nothing is imputed.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from data_pipeline.features import extract_features
from data_pipeline.gpu_export import write_gpu_input
from data_pipeline.loader import load_matrix
from gpu_bench import aggregate as agg
from gpu_bench.environment import gpu_telemetry

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXE = ROOT / "gpu" / "build" / ("spmv_bench.exe" if os.name == "nt" else "spmv_bench")
KERNELS = ("csr_scalar", "csr_vector", "csr_adaptive", "cusparse")


class BenchmarkError(RuntimeError):
    pass


def exe_path(exe: Optional[Path] = None) -> Path:
    p = Path(exe) if exe else DEFAULT_EXE
    if not p.is_file():
        raise BenchmarkError(
            f"benchmark executable not found: {p}. Build it with: python scripts/build_gpu.py"
        )
    return p


def cuda_env() -> Dict[str, str]:
    """Environment for the child process with the CUDA runtime DLL directories on PATH.

    CUDA 13 on Windows keeps the DLLs in ``bin/x64``; a shell opened before the toolkit
    was installed does not have it on PATH (the child would die with 0xC0000135).
    """
    env = dict(os.environ)
    root = env.get("CUDA_PATH")
    if not root:
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        found = sorted(base.glob("v*")) if base.is_dir() else []
        root = str(found[-1]) if found else None
    if root:
        extra = os.pathsep.join([str(Path(root) / "bin" / "x64"), str(Path(root) / "bin")])
        env["PATH"] = extra + os.pathsep + env.get("PATH", "")
    return env


def run_one(exe: Path, bin_path: Path, matrix_id: str, part_dir: Path, warmup: int, reps: int,
            eps: float, kernels: Sequence[str] = KERNELS, timing: str = "graph",
            inject_fault: str = "", timeout: float = 900.0) -> subprocess.CompletedProcess:
    part_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), "--matrix", str(bin_path), "--id", matrix_id, "--out", str(part_dir),
           "--warmup", str(warmup), "--reps", str(reps), "--eps", repr(eps),
           "--kernels", ",".join(kernels), "--timing", timing]
    if inject_fault:
        cmd += ["--inject-fault", inject_fault]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=cuda_env())


def benchmark_matrices(
    raw_dir: Path,
    matrix_ids: Sequence[str],
    out_dir: Path,
    exe: Optional[Path] = None,
    warmup: int = 20,
    reps: int = 100,
    eps: float = 1e-5,
    kernels: Sequence[str] = KERNELS,
    timing: str = "graph",
    cache_dir: Optional[Path] = None,
    file_map: Optional[Dict[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Benchmark ``matrix_ids`` (files ``raw_dir/<id>.mtx`` unless ``file_map`` given)."""
    exe = exe_path(exe)
    cache_dir = cache_dir or (out_dir / "_inputs")
    part_dir = out_dir / "_parts"
    timing_rows: List[dict] = []
    summary_rows: List[dict] = []
    failure_rows: List[dict] = []
    aux_rows: List[dict] = []
    raw_frames: List[pd.DataFrame] = []
    host_rows: List[dict] = []
    telemetry_rows: List[dict] = []

    for i, mid in enumerate(matrix_ids):
        path = (file_map or {}).get(mid, raw_dir / f"{mid}.mtx")
        try:
            t0 = time.perf_counter()
            csr, _ = load_matrix(path, matrix_id=mid)
            t1 = time.perf_counter()
            extract_features(csr, mid)
            t2 = time.perf_counter()
            binp = write_gpu_input(csr, cache_dir / f"{mid}.bin")
            t3 = time.perf_counter()
        except Exception as exc:  # recorded, never hidden
            failure_rows.append({"matrix_id": mid, "kernel_name": "*", "stage": "load_export", "reason": str(exc)[:300]})
            continue
        host_rows.append({"matrix_id": mid, "load_validate_ms": (t1 - t0) * 1e3,
                          "feature_extract_python_ms": (t2 - t1) * 1e3, "export_ms": (t3 - t2) * 1e3})
        tele0 = gpu_telemetry()
        res = run_one(exe, binp, mid, part_dir, warmup, reps, eps, kernels, timing)
        tele1 = gpu_telemetry()
        telemetry_rows.append({"matrix_id": mid, **{f"start_{k}": v for k, v in tele0.items()},
                               **{f"end_{k}": v for k, v in tele1.items()}})
        if res.returncode != 0:
            reason = {4: "insufficient_gpu_memory"}.get(res.returncode, "runner_exit_%d" % res.returncode)
            failure_rows.append({"matrix_id": mid, "kernel_name": "*", "stage": "runner",
                                 "reason": f"{reason}: {res.stderr.strip()[:250]}"})
            if verbose:
                print(f"[{i+1}/{len(matrix_ids)}] {mid}: FAILED ({reason})", flush=True)
            continue
        t, s, f, aux = agg.aggregate_matrix(mid, part_dir)
        timing_rows += t; summary_rows += s; failure_rows += f
        aux_rows.append(agg.aux_row(mid, aux))
        raw = pd.read_csv(part_dir / f"{mid}.raw.csv")
        raw.insert(0, "matrix_id", mid)
        raw_frames.append(raw)
        if verbose:
            best = min(t, key=lambda r: r["kernel_time_ms"]) if t else None
            print(f"[{i+1}/{len(matrix_ids)}] {mid}: best={best['kernel_name'] if best else '-'} "
                  f"{best['kernel_time_ms']*1e3 if best else float('nan'):.1f}us", flush=True)
        # the binary input is a cache; remove it to keep disk usage bounded
        try:
            binp.unlink()
        except OSError:
            pass

    agg.write_frames(out_dir, timing_rows, summary_rows, failure_rows, aux_rows, raw_frames)
    pd.DataFrame(host_rows).to_csv(out_dir / "host_stage_timings.csv", index=False)
    pd.DataFrame(telemetry_rows).to_csv(out_dir / "gpu_telemetry.csv", index=False)
    return {"timings": pd.read_csv(out_dir / "kernel_timings.csv"),
            "failures": pd.read_csv(out_dir / "benchmark_failures.csv")}
