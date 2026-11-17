"""
environment.py
==============
Environment capture and GPU telemetry (read-only; no clock/power settings are changed).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_TELEMETRY_FIELDS = ["temperature.gpu", "clocks.sm", "clocks.mem", "utilization.gpu",
                     "memory.used", "power.draw", "pstate"]


def _run(cmd: List[str], timeout: float = 20.0) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpu_telemetry() -> Dict[str, Optional[str]]:
    """One nvidia-smi sample. Missing values are None (never invented)."""
    out = _run(["nvidia-smi", f"--query-gpu={','.join(_TELEMETRY_FIELDS)}", "--format=csv,noheader,nounits"])
    vals = [v.strip() for v in out.splitlines()[0].split(",")] if out else []
    rec = {f: (vals[i] if i < len(vals) else None) for i, f in enumerate(_TELEMETRY_FIELDS)}
    reasons = _run(["nvidia-smi", "--query-gpu=clocks_throttle_reasons.active", "--format=csv,noheader"])
    if reasons is None:
        reasons = _run(["nvidia-smi", "--query-gpu=clocks_event_reasons.active", "--format=csv,noheader"])
    rec["throttle_reasons_active"] = reasons.splitlines()[0].strip() if reasons else None
    return rec


def git_info(root: Path) -> Dict[str, Optional[str]]:
    def g(*a: str) -> Optional[str]:
        return _run(["git", "-C", str(root), *a])
    status = g("status", "--porcelain")
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def capture_environment(root: Path, exe: Optional[Path] = None) -> Dict[str, object]:
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits"])
    gpu: Dict[str, object] = {}
    if smi:
        n, drv, mem, cc = [x.strip() for x in smi.splitlines()[0].split(",")]
        gpu = {"name": n, "driver_version": drv, "memory_total_mib": float(mem), "compute_capability": cc}
    nvcc = shutil.which("nvcc") or (str(Path(os.environ["CUDA_PATH"]) / "bin" / "nvcc.exe") if "CUDA_PATH" in os.environ else None)
    nvcc_ver = None
    if nvcc and Path(nvcc).exists():
        o = _run([nvcc, "--version"])
        nvcc_ver = o.splitlines()[-2].strip() if o else None
    versions = {}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "yaml"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": versions,
        "gpu": gpu,
        "cuda_path": os.environ.get("CUDA_PATH"),
        "nvcc": nvcc_ver,
        "benchmark_executable": str(exe) if exe else None,
        "git": git_info(root),
        "telemetry_at_start": gpu_telemetry(),
        "note": "GPU clocks and power limits are never modified by this project.",
    }


def write_environment(path: Path, env: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env, indent=2), encoding="utf-8")
