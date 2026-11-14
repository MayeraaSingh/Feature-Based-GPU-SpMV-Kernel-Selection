"""
GPU correctness / harness tests.  These run the real CUDA executable on the real GPU.
They are SKIPPED (with an explicit reason) when the executable or a CUDA device is
missing; they are never replaced by CPU emulation.  The matrices below are small
generated edge cases used ONLY for software validation, not for any reported
experiment.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from data_pipeline.gpu_export import write_gpu_input
from gpu_bench import aggregate as agg
from gpu_bench import runner

EXE = runner.DEFAULT_EXE


def _has_gpu() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and "GPU" in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not EXE.is_file() or not _has_gpu(),
    reason="CUDA executable not built (python scripts/build_gpu.py) or no NVIDIA GPU visible",
)


def _rand(rows, cols, density, seed):
    return sp.random(rows, cols, density=density, format="csr", random_state=seed, dtype=np.float64)


def _long_row():
    rng = np.random.default_rng(3)
    m = sp.lil_matrix((50, 300000))
    cols = rng.choice(300000, size=200000, replace=False)
    m[7, cols] = rng.standard_normal(cols.size)
    m[8, :5] = 1.0
    return m.tocsr()


def _power_law():
    rng = np.random.default_rng(5)
    lens = np.minimum((rng.pareto(1.2, 3000) * 2).astype(int), 2500)
    rows, cols, vals = [], [], []
    for r, L in enumerate(lens):
        c = rng.choice(4000, size=min(L, 4000), replace=False)
        rows += [r] * len(c); cols += list(c); vals += list(rng.standard_normal(len(c)))
    return sp.csr_matrix((vals, (rows, cols)), shape=(3000, 4000))


CASES = {
    "random_square": lambda: _rand(500, 500, 0.02, 1),
    "empty_rows": lambda: sp.vstack([_rand(100, 80, 0.1, 2), sp.csr_matrix((50, 80)), _rand(30, 80, 0.3, 4)]).tocsr(),
    "single_element_rows": lambda: sp.csr_matrix((np.arange(1, 65, dtype=float), (np.arange(64), np.arange(64)[::-1])), shape=(64, 64)),
    "wide_rectangular": lambda: _rand(40, 5000, 0.05, 6),
    "tall_rectangular": lambda: _rand(5000, 40, 0.1, 7),
    "extremely_long_row": _long_row,
    "power_law_irregular": _power_law,
    "diagonal_1x1": lambda: sp.csr_matrix(np.array([[2.5]])),
}


def _bench(tmp_path: Path, csr, name: str, **kw):
    binp = write_gpu_input(csr, tmp_path / f"{name}.bin")
    part = tmp_path / "parts"
    res = runner.run_one(EXE, binp, name, part, warmup=3, reps=5, eps=1e-5, **kw)
    assert res.returncode == 0, res.stderr
    return agg.aggregate_matrix(name, part)


@pytest.mark.parametrize("case", sorted(CASES))
def test_all_kernels_pass_on_edge_cases(tmp_path, case):
    timing, summary, failures, aux = _bench(tmp_path, CASES[case](), case)
    assert failures == [], failures
    by = {r["kernel_name"]: r for r in timing}
    assert set(by) == set(runner.KERNELS)
    for k, r in by.items():
        assert r["correctness_pass"], f"{k} failed correctness on {case}: {[s for s in summary if s['kernel_name']==k]}"
        assert r["kernel_time_ms"] > 0 and r["preprocessing_time_ms"] >= 0


def test_fault_injection_is_detected(tmp_path):
    timing, _, _, _ = _bench(tmp_path, CASES["random_square"](), "fault", inject_fault="csr_vector")
    by = {r["kernel_name"]: r["correctness_pass"] for r in timing}
    assert by["csr_vector"] is False
    assert all(v for k, v in by.items() if k != "csr_vector")


def test_graph_and_direct_timing_modes_agree_on_correctness(tmp_path):
    t1, s1, _, _ = _bench(tmp_path / "g", CASES["power_law_irregular"](), "m", timing="graph")
    t2, s2, _, _ = _bench(tmp_path / "d", CASES["power_law_irregular"](), "m", timing="direct")
    assert {r["kernel_name"]: r["correctness_pass"] for r in t1} == {r["kernel_name"]: r["correctness_pass"] for r in t2}
    assert {r["timing_mode"] for r in s1} == {"graph"} and {r["timing_mode"] for r in s2} == {"direct"}


def test_overhead_measurements_recorded(tmp_path):
    _, _, _, aux = _bench(tmp_path, CASES["random_square"](), "ov")
    row = agg.aux_row("ov", aux)
    assert row["feature_gpu_matches_cpu"] == "1"
    for key in ("alloc_ms", "h2d_ms", "feature_cpu_ms", "feature_gpu_ms", "timer_floor_ms"):
        assert row[key] >= 0 and np.isfinite(row[key])


def test_missing_input_file_gives_clear_error(tmp_path):
    res = runner.run_one(EXE, tmp_path / "nope.bin", "x", tmp_path / "p", 1, 1, 1e-5)
    assert res.returncode == 3 and "input error" in res.stderr


def test_missing_executable_message(tmp_path):
    with pytest.raises(runner.BenchmarkError, match="build_gpu.py"):
        runner.exe_path(tmp_path / "missing.exe")


@pytest.mark.parametrize("case", ["empty_rows", "extremely_long_row", "power_law_irregular"])
def test_multilaunch_adaptive_variant_is_also_correct(tmp_path, case):
    """Design-iteration-1 variant (kept for the documented comparison) must also be correct."""
    binp = write_gpu_input(CASES[case](), tmp_path / f"{case}.bin")
    part = tmp_path / "parts"
    res = runner.run_one(EXE, binp, case, part, warmup=3, reps=5, eps=1e-5,
                         kernels=("csr_adaptive", "csr_adaptive_multilaunch"))
    assert res.returncode == 0, res.stderr
    timing, _, failures, _ = agg.aggregate_matrix(case, part)
    assert failures == [] and all(r["correctness_pass"] for r in timing) and len(timing) == 2
