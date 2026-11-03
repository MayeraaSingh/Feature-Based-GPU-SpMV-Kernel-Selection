"""
data_pipeline
=============
Member 2 data pipeline package.

Provides:
  - Matrix loading (Matrix Market .mtx, NumPy .npz, SciPy sparse .npz)
  - CSR validation
  - Structural feature extraction
  - CPU reference SpMV  (y = A x)
  - Split-manifest generation
  - Catalog / metadata CSV writers

All public symbols are re-exported here for convenience.
"""

from data_pipeline.loader import load_matrix, load_directory
from data_pipeline.validator import validate_csr, ValidationError
from data_pipeline.features import extract_features
from data_pipeline.reference_spmv import cpu_spmv, reference_vector
from data_pipeline.splits import make_splits
from data_pipeline.catalog import write_catalog, write_features, write_splits

__all__ = [
    "load_matrix",
    "load_directory",
    "validate_csr",
    "ValidationError",
    "extract_features",
    "cpu_spmv",
    "reference_vector",
    "make_splits",
    "write_catalog",
    "write_features",
    "write_splits",
]
