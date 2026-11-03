"""
catalog.py
==========
Write the three shared metadata CSV files for the Member 2 data pipeline.

Files produced
--------------
  data/metadata/matrix_catalog.csv  — matrix_id, rows, cols, nnz, matrix_group
  data/metadata/features.csv        — matrix_id, rows, cols, nnz,
                                       mean_row_nnz, variance_row_nnz,
                                       max_row_nnz, min_row_nnz
  data/metadata/splits.csv          — matrix_id, split

The exact headers match the shared contracts in docs/data_dictionary.md.
Existing files are overwritten; the pipeline is designed to be re-runnable.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import scipy.sparse as sp

from data_pipeline.features import FeatureRecord

logger = logging.getLogger(__name__)

# Exact column order for each CSV (must match data_dictionary.md)
CATALOG_HEADERS = ["matrix_id", "rows", "cols", "nnz", "matrix_group"]
FEATURES_HEADERS = [
    "matrix_id",
    "rows",
    "cols",
    "nnz",
    "mean_row_nnz",
    "variance_row_nnz",
    "max_row_nnz",
    "min_row_nnz",
]
SPLITS_HEADERS = ["matrix_id", "split"]


def write_catalog(
    matrices: Dict[str, sp.csr_matrix],
    groups: Dict[str, str],
    output_path: Path | str,
) -> None:
    """Write ``matrix_catalog.csv``.

    Parameters
    ----------
    matrices : dict
        matrix_id → validated CSR matrix.
    groups : dict
        matrix_id → matrix_group string.
    output_path : Path or str
        Destination file path (will be created/overwritten).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_data: List[Dict] = []
    for mid in sorted(matrices):
        csr = matrices[mid]
        rows_data.append(
            {
                "matrix_id": mid,
                "rows": csr.shape[0],
                "cols": csr.shape[1],
                "nnz": csr.nnz,
                "matrix_group": groups.get(mid, "unknown"),
            }
        )

    _write_csv(output_path, CATALOG_HEADERS, rows_data)
    logger.info("Catalog written: %s (%d entries)", output_path, len(rows_data))


def write_features(
    feature_records: List[FeatureRecord],
    output_path: Path | str,
) -> None:
    """Write ``features.csv``.

    Parameters
    ----------
    feature_records : list of dict
        Each dict is the output of ``extract_features``.
    output_path : Path or str
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_data = [
        {h: rec[h] for h in FEATURES_HEADERS}
        for rec in sorted(feature_records, key=lambda r: r["matrix_id"])
    ]
    _write_csv(output_path, FEATURES_HEADERS, rows_data)
    logger.info("Features written: %s (%d entries)", output_path, len(rows_data))


def write_splits(
    manifest: Dict[str, str],
    output_path: Path | str,
) -> None:
    """Write ``splits.csv``.

    Parameters
    ----------
    manifest : dict
        matrix_id → split label.
    output_path : Path or str
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_data = [
        {"matrix_id": mid, "split": label}
        for mid, label in sorted(manifest.items())
    ]
    _write_csv(output_path, SPLITS_HEADERS, rows_data)
    logger.info("Splits written: %s (%d entries)", output_path, len(rows_data))


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _write_csv(path: Path, headers: List[str], rows: List[Dict]) -> None:
    """Write a CSV file with the exact given headers, overwriting if present."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_csv_headers(path: Path | str, expected_headers: List[str]) -> bool:
    """Return True if the CSV at ``path`` has exactly the expected headers.

    Useful for contract compliance checks in tests and integration.
    """
    path = Path(path)
    if not path.is_file():
        return False
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            actual = next(reader)
        except StopIteration:
            return False
    return actual == expected_headers
