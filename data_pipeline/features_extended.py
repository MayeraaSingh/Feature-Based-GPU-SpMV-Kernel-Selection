"""
features_extended.py
====================
Additional row-distribution features, written to a SEPARATE file
(``features_extended.csv``) so the original 7-feature contract in
``docs/data_dictionary.md`` (``features.csv``) is unchanged.

Only four extra features are defined (not every conceivable statistic), each answering a
specific question about CSR-kernel behaviour:

  cv_row_nnz        sqrt(variance_row_nnz) / mean_row_nnz     scale-free irregularity
  empty_row_frac    fraction of rows with zero nonzeros        wasted threads / pure-write rows
  max_over_mean     max_row_nnz / mean_row_nnz                 heavy-tail / imbalance indicator
  frac_rows_gt_32   fraction of rows with more than 32 nnz     rows that can fill a warp
                                                               (32 = warp width, fixed a priori)

All are computed from the CSR row-pointer array only, i.e. they are available before any
GPU timing exists and never depend on timings.  ``std`` is omitted (it is a monotone
function of the existing variance) and percentiles are omitted (highly correlated with
max/mean on this data; the ablation experiment tests whether the four above help at all).
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import scipy.sparse as sp

EXTENDED_COLUMNS = ["cv_row_nnz", "empty_row_frac", "max_over_mean", "frac_rows_gt_32"]
WARP = 32


def extract_extended_features(csr: sp.csr_matrix, matrix_id: str) -> Dict[str, Union[str, float]]:
    if not sp.isspmatrix_csr(csr):
        raise TypeError(f"[{matrix_id}] Expected a CSR matrix; got {type(csr).__name__}.")
    counts = np.diff(csr.indptr).astype(np.int64)
    rows = counts.size
    mean = float(counts.mean()) if rows else 0.0
    var = float(counts.var()) if rows else 0.0
    mx = int(counts.max()) if rows else 0
    return {
        "matrix_id": matrix_id,
        "cv_row_nnz": round(float(np.sqrt(var) / mean), 6) if mean > 0 else 0.0,
        "empty_row_frac": round(float((counts == 0).mean()), 6) if rows else 0.0,
        "max_over_mean": round(float(mx / mean), 6) if mean > 0 else 0.0,
        "frac_rows_gt_32": round(float((counts > WARP).mean()), 6) if rows else 0.0,
    }
