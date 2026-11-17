"""
check_environment.py
====================
Verify (read-only) that this machine can run the GPU experiment and print a report:
GPU / driver / compute capability, CUDA toolkit, MSVC / CMake / Ninja (Windows), Python packages,
whether the benchmark executable is built.  Exit code 0 only when everything required for GPU
benchmarking is present; CPU-only work (selector training, evaluation) needs just the Python packages.

    python scripts/check_environment.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from gpu_bench.environment import capture_environment
from gpu_bench.runner import DEFAULT_EXE

REQUIRED_PACKAGES = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "yaml"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    env = capture_environment(_root, DEFAULT_EXE if DEFAULT_EXE.is_file() else None)
    problems = []

    print(f"Python        : {env['python']}  ({env['platform']})")
    for k, v in env["packages"].items():
        print(f"  {k:<11}: {v or 'MISSING'}")
        if v is None and k in REQUIRED_PACKAGES:
            problems.append(f"python package missing: {k} (pip install -r requirements.txt)")
    gpu = env["gpu"]
    if gpu:
        print(f"GPU           : {gpu['name']}  cc {gpu['compute_capability']}  {gpu['memory_total_mib']:.0f} MiB  driver {gpu['driver_version']}")
    else:
        problems.append("no NVIDIA GPU / nvidia-smi not available (GPU benchmarking impossible; CPU-only work still fine)")
    nvcc = env["nvcc"]
    print(f"nvcc          : {nvcc or 'NOT FOUND'}")
    if not nvcc:
        problems.append("CUDA Toolkit / nvcc not found (set CUDA_PATH; see docs/environment_setup.md)")
    if os.name == "nt":
        vs = any(Path(p).exists() for p in (r"C:\Program Files\Microsoft Visual Studio", r"C:\Program Files (x86)\Microsoft Visual Studio"))
        print(f"Visual Studio : {'found' if vs else 'NOT FOUND'}")
        if not vs:
            problems.append("Visual Studio / MSVC build tools not found")
    else:
        for tool in ("cmake", "g++"):
            if shutil.which(tool) is None:
                problems.append(f"{tool} not found")
    exe_ok = DEFAULT_EXE.is_file()
    print(f"spmv_bench    : {'built (' + str(DEFAULT_EXE) + ')' if exe_ok else 'NOT BUILT  ->  python scripts/build_gpu.py'}")
    if not exe_ok:
        problems.append("benchmark executable not built")
    print(f"git           : {env['git']}")
    if args.json:
        Path(args.json).write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("\nREADY for GPU experiments." if not problems else "\nPROBLEMS:\n  - " + "\n  - ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
