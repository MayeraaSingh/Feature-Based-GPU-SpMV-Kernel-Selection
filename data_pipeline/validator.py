"""
validator.py
============
CSR structural validation for the Member 2 data pipeline.

Policy (documented here and in data_pipeline/README.md):
  - Duplicate entries: summed into a single entry (via scipy eliminate_zeros /
    sum_duplicates).  Duplicates are logged as warnings, not hard errors, because
    many real-world Matrix Market files legitimately encode symmetric/Hermitian
    matrices using duplicate (i, j) pairs.
  - Explicit zeros: removed after duplicate summation (some duplicates cancel to
    zero).  The post-removal nnz is the canonical nnz stored in the catalog.
  - Index type: col_idx values are stored as int32 (or int64 for very large
    matrices, i.e. rows or cols > 2^31−1).  The validator always promotes to
    int64 for range checking, then stores back as int32 unless necessary.
  - Empty rows: rows with zero nonzeros are valid; they contribute 0 to
    per-row counts, 0 to min_row_nnz (unless the matrix is fully empty, which
    is an error), and are included in variance/mean calculations.

Variance convention:
  Population variance is used (ddof=0), consistent with NumPy's default.
  This is documented in the data dictionary.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a matrix fails a hard CSR validation check."""


def validate_csr(matrix: sp.spmatrix, matrix_id: str = "<unknown>") -> sp.csr_matrix:
    """Validate and canonicalize a sparse matrix to CSR.

    Steps performed (in order):
    1. Convert to CSR.
    2. Sort indices within each row (required by cuSPARSE and for determinism).
    3. Sum duplicate (i, j) entries.
    4. Remove explicit zeros.
    5. Validate dimensions, pointer lengths, index ranges, and nnz consistency.

    Parameters
    ----------
    matrix : sp.spmatrix
        Any SciPy sparse matrix.
    matrix_id : str
        Human-readable identifier used in log / error messages.

    Returns
    -------
    sp.csr_matrix
        A canonical CSR matrix with:
        - sorted column indices per row,
        - no duplicate (i, j) entries,
        - no explicit zeros,
        - int32 indices (int64 if dimensions exceed int32 range).

    Raises
    ------
    ValidationError
        For any structural integrity failure that cannot be silently corrected.
    """
    # ------------------------------------------------------------------ #
    # 1. Convert to CSR                                                    #
    # ------------------------------------------------------------------ #
    if not sp.issparse(matrix):
        raise ValidationError(f"[{matrix_id}] Input is not a SciPy sparse matrix.")

    csr = matrix.tocsr()
    rows, cols = csr.shape

    if rows <= 0 or cols <= 0:
        raise ValidationError(
            f"[{matrix_id}] Invalid dimensions: ({rows}, {cols}). "
            "Both rows and cols must be positive."
        )

    # ------------------------------------------------------------------ #
    # 2. Sort indices (in-place)                                           #
    # ------------------------------------------------------------------ #
    csr.sort_indices()

    # ------------------------------------------------------------------ #
    # 3. Sum duplicates                                                    #
    # ------------------------------------------------------------------ #
    nnz_before = csr.nnz
    csr.sum_duplicates()
    if csr.nnz != nnz_before:
        logger.warning(
            "[%s] Duplicate (i, j) entries detected and summed. "
            "nnz changed %d → %d.",
            matrix_id,
            nnz_before,
            csr.nnz,
        )

    # ------------------------------------------------------------------ #
    # 4. Remove explicit zeros                                             #
    # ------------------------------------------------------------------ #
    nnz_before_elim = csr.nnz
    csr.eliminate_zeros()
    if csr.nnz != nnz_before_elim:
        logger.warning(
            "[%s] Explicit zeros removed. nnz changed %d → %d.",
            matrix_id,
            nnz_before_elim,
            csr.nnz,
        )

    nnz = csr.nnz

    # ------------------------------------------------------------------ #
    # 5. Structural integrity checks                                       #
    # ------------------------------------------------------------------ #

    # 5a. row_ptr length
    expected_ptr_len = rows + 1
    actual_ptr_len = len(csr.indptr)
    if actual_ptr_len != expected_ptr_len:
        raise ValidationError(
            f"[{matrix_id}] row_ptr (indptr) length mismatch: "
            f"expected {expected_ptr_len}, got {actual_ptr_len}."
        )

    # 5b. row_ptr starts at 0
    if csr.indptr[0] != 0:
        raise ValidationError(
            f"[{matrix_id}] row_ptr[0] = {csr.indptr[0]}; must be 0."
        )

    # 5c. row_ptr terminates at nnz
    if csr.indptr[-1] != nnz:
        raise ValidationError(
            f"[{matrix_id}] row_ptr[-1] = {csr.indptr[-1]}; expected nnz = {nnz}."
        )

    # 5d. row_ptr is monotonically non-decreasing
    diffs = np.diff(csr.indptr.astype(np.int64))
    if np.any(diffs < 0):
        raise ValidationError(
            f"[{matrix_id}] row_ptr (indptr) is not monotonically non-decreasing."
        )

    # 5e. col_idx length agrees with nnz
    if len(csr.indices) != nnz:
        raise ValidationError(
            f"[{matrix_id}] col_idx (indices) length = {len(csr.indices)}; "
            f"expected nnz = {nnz}."
        )

    # 5f. col_idx values in range [0, cols)
    if nnz > 0:
        col_min = int(csr.indices.min())
        col_max = int(csr.indices.max())
        if col_min < 0:
            raise ValidationError(
                f"[{matrix_id}] Negative column index found: {col_min}."
            )
        if col_max >= cols:
            raise ValidationError(
                f"[{matrix_id}] Column index {col_max} out of range [0, {cols})."
            )

    # 5g. values array length
    if len(csr.data) != nnz:
        raise ValidationError(
            f"[{matrix_id}] values array length = {len(csr.data)}; "
            f"expected nnz = {nnz}."
        )

    # 5h. No NaN or Inf in values
    if nnz > 0 and not np.all(np.isfinite(csr.data)):
        n_bad = int(np.sum(~np.isfinite(csr.data)))
        raise ValidationError(
            f"[{matrix_id}] {n_bad} non-finite value(s) (NaN/Inf) in data array."
        )

    # ------------------------------------------------------------------ #
    # 6. Promote index dtype if needed                                     #
    # ------------------------------------------------------------------ #
    int32_max = np.iinfo(np.int32).max
    if rows > int32_max or cols > int32_max:
        csr = csr.astype(csr.dtype, copy=False)
        csr.indices = csr.indices.astype(np.int64)
        csr.indptr = csr.indptr.astype(np.int64)
        logger.info(
            "[%s] Dimension exceeds int32 range; indices promoted to int64.",
            matrix_id,
        )
    else:
        csr.indices = csr.indices.astype(np.int32, copy=False)
        csr.indptr = csr.indptr.astype(np.int32, copy=False)

    logger.info(
        "[%s] Validation passed: shape=(%d, %d), nnz=%d.",
        matrix_id,
        rows,
        cols,
        nnz,
    )
    return csr
