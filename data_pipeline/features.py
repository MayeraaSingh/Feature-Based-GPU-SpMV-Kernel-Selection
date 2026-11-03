"""
features.py
===========
Structural feature extraction for the Member 2 data pipeline.

Agreed features (from docs/data_dictionary.md and project_context_scope.md):

  rows            Number of matrix rows
  cols            Number of matrix columns
  nnz             Number of nonzero entries in the standardised CSR matrix
  mean_row_nnz    Mean of per-row nonzero counts  = nnz / rows
  variance_row_nnz  Population variance of per-row nonzero counts (ddof=0)
  max_row_nnz     Maximum per-row nonzero count
  min_row_nnz     Minimum per-row nonzero count

Conventions
-----------
1. **Source of row counts**: derived from the CSR row-pointer array (indptr).
   row_counts[i] = indptr[i+1] - indptr[i]  for i in 0..rows-1
   This is the canonical definition from the CSR representation and does not
   rely on values or indices arrays.

2. **Variance convention**: *population variance* (ddof=0, NumPy default).
   For a matrix with all equal row lengths this gives 0.0, which is the
   expected result.  Sample variance (ddof=1) would be undefined for a
   single-row matrix.  The choice of population variance is recorded here and
   in the data dictionary.

3. **Empty rows**: rows with zero nonzeros are valid and included in all
   statistics.  They contribute 0 to mean (via nnz count), increase variance
   when other rows are non-empty, set min_row_nnz=0 if any such row exists,
   and are never omitted.

4. **Fully-empty matrix** (nnz=0): all statistics are defined—mean=0.0,
   variance=0.0, max=0, min=0.  The pipeline logs a warning for such matrices.
"""

from __future__ import annotations

import logging
from typing import Dict, Union

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# Type for a single feature record
FeatureRecord = Dict[str, Union[str, int, float]]


def extract_features(csr: sp.csr_matrix, matrix_id: str) -> FeatureRecord:
    """Extract all agreed structural features from a validated CSR matrix.

    Parameters
    ----------
    csr : sp.csr_matrix
        A validated CSR matrix (output of ``validate_csr``).
    matrix_id : str
        Stable matrix identifier to include in the returned record.

    Returns
    -------
    dict with keys:
        matrix_id, rows, cols, nnz,
        mean_row_nnz, variance_row_nnz, max_row_nnz, min_row_nnz
    """
    if not sp.isspmatrix_csr(csr):
        raise TypeError(
            f"[{matrix_id}] Expected a CSR matrix; got {type(csr).__name__}."
        )

    rows, cols = csr.shape
    nnz = csr.nnz

    # Per-row nonzero counts derived from the row-pointer array
    row_counts = np.diff(csr.indptr).astype(np.int64)  # shape: (rows,)

    if nnz == 0:
        logger.warning(
            "[%s] Matrix is fully empty (nnz=0). "
            "All row-nnz statistics will be 0.",
            matrix_id,
        )

    mean_row_nnz: float = float(np.mean(row_counts))
    variance_row_nnz: float = float(np.var(row_counts, ddof=0))  # population variance
    max_row_nnz: int = int(np.max(row_counts)) if rows > 0 else 0
    min_row_nnz: int = int(np.min(row_counts)) if rows > 0 else 0

    record: FeatureRecord = {
        "matrix_id": matrix_id,
        "rows": int(rows),
        "cols": int(cols),
        "nnz": int(nnz),
        "mean_row_nnz": round(mean_row_nnz, 6),
        "variance_row_nnz": round(variance_row_nnz, 6),
        "max_row_nnz": max_row_nnz,
        "min_row_nnz": min_row_nnz,
    }

    logger.info(
        "[%s] Features: rows=%d, cols=%d, nnz=%d, "
        "mean=%.4f, var=%.4f, max=%d, min=%d",
        matrix_id,
        rows,
        cols,
        nnz,
        mean_row_nnz,
        variance_row_nnz,
        max_row_nnz,
        min_row_nnz,
    )
    return record
