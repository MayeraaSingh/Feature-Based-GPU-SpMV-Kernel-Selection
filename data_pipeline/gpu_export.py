"""
gpu_export.py
=============
Serialise a validated CSR matrix, the input vector and the float64 CPU reference into
the binary format read by ``gpu/benchmarks/benchmark_runner.cu``.

Layout (little endian)::

    char[8]  magic = "SPMVBIN1"
    int64    rows, cols, nnz
    float64  norm_a_inf   (max absolute row sum of the float64 matrix)
    float64  norm_x_inf   (max |x|, x being the float32-rounded vector)
    int32    indptr[rows+1]
    int32    indices[nnz]
    float32  values[nnz]        (float32 rounding of the float64 values)
    float32  x[cols]            (mixed-sign, see reference_spmv.reference_vector)
    float64  y_ref[rows]        (CPU float64 SpMV of the float64 matrix with the same x)

The reference uses the ORIGINAL float64 matrix values, so the correctness check also
covers the float32 rounding of the matrix values, exactly as the benchmark protocol
states (float64 CPU reference vs float32 GPU computation).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from data_pipeline.reference_spmv import cpu_spmv, reference_vector

MAGIC = b"SPMVBIN1"
INT32_MAX = 2**31 - 1


def make_problem(csr: sp.csr_matrix, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return (x_float32, y_ref_float64, norm_a_inf, norm_x_inf)."""
    cols = csr.shape[1]
    x32 = reference_vector(cols, seed=seed, dtype=np.float32, mixed_sign=True)
    y_ref, x_used = cpu_spmv(csr, x=x32.astype(np.float64))
    norm_a = float(np.abs(csr).sum(axis=1).max()) if csr.nnz else 0.0
    norm_x = float(np.abs(x32).max()) if cols else 0.0
    return x32, y_ref, norm_a, norm_x


def write_gpu_input(csr: sp.csr_matrix, path: Path | str, seed: int = 42) -> Path:
    """Write ``csr`` to ``path`` in the GPU benchmark format."""
    rows, cols = csr.shape
    if csr.nnz > INT32_MAX or rows > INT32_MAX or cols > INT32_MAX:
        raise ValueError("matrix dimensions/nnz exceed int32 range supported by the GPU kernels")
    x32, y_ref, norm_a, norm_x = make_problem(csr, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<qqq", rows, cols, csr.nnz))
        fh.write(struct.pack("<dd", norm_a, norm_x))
        fh.write(csr.indptr.astype("<i4").tobytes())
        fh.write(csr.indices.astype("<i4").tobytes())
        fh.write(csr.data.astype("<f4").tobytes())
        fh.write(x32.astype("<f4").tobytes())
        fh.write(y_ref.astype("<f8").tobytes())
    return path
