"""
pilot_data.py
=============
Synthetic pilot dataset generator for the Member 2 data pipeline.

PURPOSE: Development and testing ONLY.
---------------------------------------
These matrices are NOT real-world sparse matrices and must never be
presented as the final research dataset or used as final benchmark evidence.

Each synthetic matrix is stored as a SciPy sparse matrix (.npz) in
``data/raw/`` and labelled with its generation parameters so that its
origin is unambiguous.

Synthetic matrix catalogue (7 matrices across 3 structural groups):

  pilot_diag_100       — 100×100 diagonal (1 nnz / row)          [synthetic_diagonal]
  pilot_dense_block_50 — 50×50 dense block (50 nnz / row)        [synthetic_dense]
  pilot_uniform_200    — 200×200, 10 nnz/row, uniform random     [synthetic_uniform]
  pilot_skew_150       — 150×150, heavy-tail row-length dist      [synthetic_skewed]
  pilot_banded_80      — 80×80, bandwidth-5 banded               [synthetic_banded]
  pilot_empty_rows_60  — 60×60, 30% empty rows                   [synthetic_sparse]
  pilot_large_sparse   — 500×500, 3 nnz/row average              [synthetic_sparse]

These seven matrices give:
  - Three structural groups (so group-based splitting is exercisable).
  - Varied row-length distributions (uniform, diagonal, dense, skewed,
    banded, with/without empty rows).
  - A range of sizes suitable for fast unit testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

PILOT_LABEL = "SYNTHETIC_PILOT"

# (matrix_id, generator_fn_name, group)
_PILOT_REGISTRY: List[Tuple[str, str, str]] = [
    ("pilot_diag_100",       "_make_diagonal",       "synthetic_diagonal"),
    ("pilot_dense_block_50", "_make_dense_block",    "synthetic_dense"),
    ("pilot_uniform_200",    "_make_uniform_random", "synthetic_uniform"),
    ("pilot_skew_150",       "_make_skewed",         "synthetic_skewed"),
    ("pilot_banded_80",      "_make_banded",         "synthetic_banded"),
    ("pilot_empty_rows_60",  "_make_empty_rows",     "synthetic_sparse"),
    ("pilot_large_sparse",   "_make_large_sparse",   "synthetic_sparse"),
]


def generate_pilot_matrices(
    output_dir: Path | str,
    overwrite: bool = False,
) -> Dict[str, str]:
    """Generate synthetic pilot matrices and save them to ``output_dir``.

    Parameters
    ----------
    output_dir : Path or str
        Directory to write .npz files into (will be created if absent).
    overwrite : bool
        If False (default), skip files that already exist.

    Returns
    -------
    dict mapping matrix_id → matrix_group
        Suitable for passing directly to the pipeline as the ``groups`` dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, str] = {}

    for matrix_id, generator_name, group in _PILOT_REGISTRY:
        dest = output_dir / f"{matrix_id}.npz"
        groups[matrix_id] = group

        if dest.exists() and not overwrite:
            logger.info("[%s] Already exists, skipping generation: %s", matrix_id, dest)
            continue

        generator = _GENERATORS[generator_name]
        mat: sp.csr_matrix = generator()
        sp.save_npz(str(dest), mat)
        logger.info(
            "[%s] %s generated: shape=%s, nnz=%d → %s",
            PILOT_LABEL,
            matrix_id,
            mat.shape,
            mat.nnz,
            dest,
        )

    return groups


# ------------------------------------------------------------------ #
# Synthetic matrix generators                                          #
# ------------------------------------------------------------------ #

def _make_diagonal() -> sp.csr_matrix:
    """100×100 diagonal matrix — 1 nnz per row, perfectly uniform."""
    return sp.eye(100, format="csr", dtype=np.float64)


def _make_dense_block() -> sp.csr_matrix:
    """50×50 fully dense block — 50 nnz per row."""
    return sp.csr_matrix(np.ones((50, 50), dtype=np.float64))


def _make_uniform_random() -> sp.csr_matrix:
    """200×200 matrix with exactly 10 nnz per row (uniform row length)."""
    rng = np.random.default_rng(0)
    rows_idx = []
    cols_idx = []
    for r in range(200):
        cols = rng.choice(200, size=10, replace=False)
        rows_idx.extend([r] * 10)
        cols_idx.extend(cols.tolist())
    data = rng.random(len(rows_idx))
    mat = sp.csr_matrix(
        (data, (rows_idx, cols_idx)), shape=(200, 200), dtype=np.float64
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def _make_skewed() -> sp.csr_matrix:
    """150×150 matrix with heavy-tailed row-length distribution.

    Row lengths follow a power-law approximation: a small number of rows
    have many nonzeros while most rows have very few.  This exercises
    variance and max_row_nnz statistics.
    """
    rng = np.random.default_rng(1)
    n = 150
    rows_idx = []
    cols_idx = []
    for r in range(n):
        # Power-law: P(k) ∝ k^(-2) clamped to [1, n-1]
        k = int(min(n - 1, max(1, int(np.floor(n / (rng.integers(1, n + 1)))))))
        cols = rng.choice(n, size=k, replace=False)
        rows_idx.extend([r] * k)
        cols_idx.extend(cols.tolist())
    data = rng.random(len(rows_idx)) + 0.1  # avoid exact zeros
    mat = sp.csr_matrix(
        (data, (rows_idx, cols_idx)), shape=(n, n), dtype=np.float64
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def _make_banded() -> sp.csr_matrix:
    """80×80 banded matrix with bandwidth 5 (diagonals -2..+2)."""
    n = 80
    diags = [np.ones(n), np.ones(n - 1), np.ones(n - 1),
             np.ones(n - 2), np.ones(n - 2)]
    offsets = [0, 1, -1, 2, -2]
    mat = sp.diags(diags, offsets, shape=(n, n), format="csr", dtype=np.float64)
    return mat


def _make_empty_rows() -> sp.csr_matrix:
    """60×60 matrix with ~30% empty rows (18 rows have zero nonzeros)."""
    rng = np.random.default_rng(2)
    n = 60
    active_rows = sorted(rng.choice(n, size=42, replace=False).tolist())
    rows_idx = []
    cols_idx = []
    for r in active_rows:
        k = rng.integers(1, 8)
        cols = rng.choice(n, size=k, replace=False)
        rows_idx.extend([r] * int(k))
        cols_idx.extend(cols.tolist())
    data = rng.random(len(rows_idx)) + 0.1
    mat = sp.csr_matrix(
        (data, (rows_idx, cols_idx)), shape=(n, n), dtype=np.float64
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def _make_large_sparse() -> sp.csr_matrix:
    """500×500 sparse matrix, ~3 nnz per row on average."""
    rng = np.random.default_rng(3)
    n = 500
    nnz_target = n * 3
    row_idx = rng.integers(0, n, size=nnz_target)
    col_idx = rng.integers(0, n, size=nnz_target)
    data = rng.random(nnz_target) + 0.1
    mat = sp.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n, n), dtype=np.float64
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


# Registry used by generate_pilot_matrices
_GENERATORS: Dict[str, object] = {
    "_make_diagonal":       _make_diagonal,
    "_make_dense_block":    _make_dense_block,
    "_make_uniform_random": _make_uniform_random,
    "_make_skewed":         _make_skewed,
    "_make_banded":         _make_banded,
    "_make_empty_rows":     _make_empty_rows,
    "_make_large_sparse":   _make_large_sparse,
}
