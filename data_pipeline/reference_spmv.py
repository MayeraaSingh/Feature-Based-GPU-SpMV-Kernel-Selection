"""
reference_spmv.py
=================
CPU reference SpMV (y = A x) for the Member 2 data pipeline.

Purpose
-------
Provides a trusted, deterministic CPU result that Member 1 uses to verify
GPU kernel correctness.  The implementation wraps SciPy's sparse matrix-vector
product, which is well-tested and numerically reliable for float64 arithmetic.

Usage for correctness checks (documented in docs/benchmark_protocol.md)
-----------------------------------------------------------------------
The GPU benchmark protocol requires:

    ‖y_gpu − y_cpu‖_∞  ≤  ε · (‖A‖_∞ · ‖x‖_∞ + 1)

with ε = 1e-5 (float32 GPU computation) or ε = 1e-10 (float64).

Use ``check_result`` to perform this comparison.

Numerical policy
----------------
- The CPU reference always computes in float64 for maximum accuracy.
- The input vector x is constructed from a fixed random seed (or supplied by
  the caller) so that the same x is used for both the CPU and GPU runs.
- The reference vector y is reproducible given the same (A, x, seed).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Default tolerance for float32 GPU comparisons (relative to ‖A‖_∞ · ‖x‖_∞)
DEFAULT_TOLERANCE: float = 1e-5


def reference_vector(cols: int, seed: int = 42, dtype: np.dtype = np.float64) -> np.ndarray:
    """Generate the reference input vector x.

    Parameters
    ----------
    cols : int
        Number of columns in the matrix (length of x).
    seed : int
        Random seed for reproducibility.
    dtype : numpy dtype
        Data type for the output vector.

    Returns
    -------
    np.ndarray of shape (cols,) with values in [0, 1).
    """
    rng = np.random.default_rng(seed)
    return rng.random(cols).astype(dtype)


def cpu_spmv(
    csr: sp.csr_matrix,
    x: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute y = A x on the CPU using SciPy.

    Parameters
    ----------
    csr : sp.csr_matrix
        A validated CSR matrix.
    x : np.ndarray, optional
        Input vector of shape (cols,).  If None, a reproducible vector is
        generated from ``reference_vector(cols, seed)``.
    seed : int
        Seed used to generate x when x is None.

    Returns
    -------
    (y, x) : tuple
        y : np.ndarray of shape (rows,) — the product Ax in float64.
        x : np.ndarray of shape (cols,) — the input vector used.

    Notes
    -----
    The computation is performed in float64 regardless of the stored dtype of
    the matrix values, to provide a stable reference.
    """
    if not sp.issparse(csr):
        raise TypeError(f"Expected a SciPy sparse matrix; got {type(csr).__name__}.")

    rows, cols = csr.shape

    if x is None:
        x = reference_vector(cols, seed=seed, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (cols,):
            raise ValueError(
                f"Vector x has shape {x.shape}; expected ({cols},) "
                f"for a matrix with {cols} columns."
            )

    # Promote matrix to float64 for the multiplication
    A = csr.astype(np.float64)
    y = A.dot(x)

    logger.debug(
        "cpu_spmv: shape=(%d, %d), nnz=%d, ‖y‖_∞=%.6e",
        rows,
        cols,
        csr.nnz,
        float(np.max(np.abs(y))) if rows > 0 else 0.0,
    )
    return y, x


def check_result(
    y_ref: np.ndarray,
    y_test: np.ndarray,
    A: sp.spmatrix,
    x: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[bool, float, float]:
    """Check whether a test vector matches the CPU reference within tolerance.

    Uses the benchmark protocol criterion:
        ‖y_test − y_ref‖_∞  ≤  tolerance · (‖A‖_∞ · ‖x‖_∞ + 1)

    Parameters
    ----------
    y_ref : np.ndarray
        CPU reference output vector.
    y_test : np.ndarray
        GPU (or other) output vector to validate.
    A : sp.spmatrix
        The sparse matrix (used to compute ‖A‖_∞).
    x : np.ndarray
        The input vector used for both computations.
    tolerance : float
        Relative tolerance ε.

    Returns
    -------
    (passed, abs_error, threshold) : tuple
        passed     : bool — True if the check passes.
        abs_error  : float — ‖y_test − y_ref‖_∞
        threshold  : float — tolerance · (‖A‖_∞ · ‖x‖_∞ + 1)
    """
    y_ref = np.asarray(y_ref, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    if y_ref.shape != y_test.shape:
        raise ValueError(
            f"Shape mismatch: y_ref {y_ref.shape} vs y_test {y_test.shape}."
        )

    # ‖A‖_∞ = max row-sum of absolute values
    norm_A_inf = float(np.max(np.abs(A).sum(axis=1))) if A.nnz > 0 else 0.0
    norm_x_inf = float(np.max(np.abs(x))) if len(x) > 0 else 0.0
    threshold = tolerance * (norm_A_inf * norm_x_inf + 1.0)

    abs_error = float(np.max(np.abs(y_test - y_ref)))
    passed = abs_error <= threshold

    logger.debug(
        "check_result: abs_error=%.4e, threshold=%.4e, passed=%s",
        abs_error,
        threshold,
        passed,
    )
    return passed, abs_error, threshold
