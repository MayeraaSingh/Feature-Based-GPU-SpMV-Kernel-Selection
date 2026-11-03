"""
test_contracts.py
=================
Contract compliance tests for the three shared metadata CSV files.

Tests verify:
  - Correct headers matching docs/data_dictionary.md exactly.
  - Consistent matrix_id across all three files.
  - No duplicate matrix_ids within a file.
  - Split labels are drawn from {"train", "val", "test"}.
  - Feature values are finite.
  - Catalog nnz agrees with features nnz for every matrix_id.

These tests run against *programmatically written* CSV files (not on-disk
experiment artifacts) to keep them fast and self-contained.

Run with:
    pytest tests/test_contracts.py -v
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from data_pipeline.catalog import (
    write_catalog,
    write_features,
    write_splits,
    validate_csv_headers,
    CATALOG_HEADERS,
    FEATURES_HEADERS,
    SPLITS_HEADERS,
)
from data_pipeline.features import extract_features
from data_pipeline.splits import make_splits
from data_pipeline.validator import validate_csr


# =========================================================================== #
# Shared fixture: write all three CSVs to a temp dir                           #
# =========================================================================== #

@pytest.fixture
def tmp_metadata(tmp_path):
    """Build a minimal but complete set of metadata CSVs and return paths."""
    # 4 small synthetic matrices
    matrices = {}
    groups = {}
    for i in range(4):
        n = 5 + i
        mat = validate_csr(sp.eye(n, format="csr"), f"mat_{i:02d}")
        matrices[f"mat_{i:02d}"] = mat
        groups[f"mat_{i:02d}"] = f"grp_{i % 2}"

    feature_records = [
        extract_features(mat, mid) for mid, mat in sorted(matrices.items())
    ]
    manifest = make_splits(
        list(matrices.keys()),
        groups=groups,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=0,
    )

    catalog_path  = tmp_path / "matrix_catalog.csv"
    features_path = tmp_path / "features.csv"
    splits_path   = tmp_path / "splits.csv"

    write_catalog(matrices, groups, catalog_path)
    write_features(feature_records, features_path)
    write_splits(manifest, splits_path)

    return {
        "catalog":  catalog_path,
        "features": features_path,
        "splits":   splits_path,
        "matrices": matrices,
        "groups":   groups,
    }


# =========================================================================== #
# Header contract tests                                                         #
# =========================================================================== #

class TestCsvHeaders:
    def test_catalog_headers(self, tmp_metadata):
        assert validate_csv_headers(
            tmp_metadata["catalog"], CATALOG_HEADERS
        ), f"Expected headers: {CATALOG_HEADERS}"

    def test_features_headers(self, tmp_metadata):
        assert validate_csv_headers(
            tmp_metadata["features"], FEATURES_HEADERS
        ), f"Expected headers: {FEATURES_HEADERS}"

    def test_splits_headers(self, tmp_metadata):
        assert validate_csv_headers(
            tmp_metadata["splits"], SPLITS_HEADERS
        ), f"Expected headers: {SPLITS_HEADERS}"

    def test_exact_catalog_header_string(self, tmp_metadata):
        """Spot-check: first line must be exactly the agreed string."""
        with open(tmp_metadata["catalog"], "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        assert first_line == "matrix_id,rows,cols,nnz,matrix_group"

    def test_exact_features_header_string(self, tmp_metadata):
        with open(tmp_metadata["features"], "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        assert first_line == (
            "matrix_id,rows,cols,nnz,"
            "mean_row_nnz,variance_row_nnz,max_row_nnz,min_row_nnz"
        )

    def test_exact_splits_header_string(self, tmp_metadata):
        with open(tmp_metadata["splits"], "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        assert first_line == "matrix_id,split"


# =========================================================================== #
# matrix_id consistency tests                                                   #
# =========================================================================== #

def _read_ids(path: Path, id_col: str = "matrix_id") -> list:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row[id_col] for row in reader]


class TestMatrixIdConsistency:
    def test_catalog_and_features_same_ids(self, tmp_metadata):
        catalog_ids = set(_read_ids(tmp_metadata["catalog"]))
        features_ids = set(_read_ids(tmp_metadata["features"]))
        assert catalog_ids == features_ids, (
            f"Catalog IDs: {catalog_ids}\nFeature IDs: {features_ids}"
        )

    def test_catalog_and_splits_same_ids(self, tmp_metadata):
        catalog_ids = set(_read_ids(tmp_metadata["catalog"]))
        splits_ids  = set(_read_ids(tmp_metadata["splits"]))
        assert catalog_ids == splits_ids

    def test_no_duplicate_ids_in_catalog(self, tmp_metadata):
        ids = _read_ids(tmp_metadata["catalog"])
        assert len(ids) == len(set(ids)), "Duplicate matrix_ids in catalog"

    def test_no_duplicate_ids_in_features(self, tmp_metadata):
        ids = _read_ids(tmp_metadata["features"])
        assert len(ids) == len(set(ids)), "Duplicate matrix_ids in features"

    def test_no_duplicate_ids_in_splits(self, tmp_metadata):
        ids = _read_ids(tmp_metadata["splits"])
        assert len(ids) == len(set(ids)), "Duplicate matrix_ids in splits"


# =========================================================================== #
# Data quality checks                                                           #
# =========================================================================== #

class TestDataQuality:
    def test_split_labels_valid(self, tmp_metadata):
        valid_labels = {"train", "val", "test"}
        with open(tmp_metadata["splits"], "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["split"] in valid_labels, (
                    f"Invalid split label '{row['split']}' for {row['matrix_id']}"
                )

    def test_feature_values_finite(self, tmp_metadata):
        float_cols = ["mean_row_nnz", "variance_row_nnz"]
        with open(tmp_metadata["features"], "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in float_cols:
                    val = float(row[col])
                    assert np.isfinite(val), (
                        f"Non-finite value in {col} for {row['matrix_id']}: {val}"
                    )

    def test_catalog_nnz_agrees_with_features(self, tmp_metadata):
        """nnz in catalog must equal nnz in features for every matrix_id."""
        catalog_nnz = {}
        with open(tmp_metadata["catalog"], "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                catalog_nnz[row["matrix_id"]] = int(row["nnz"])
        features_nnz = {}
        with open(tmp_metadata["features"], "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                features_nnz[row["matrix_id"]] = int(row["nnz"])
        for mid in catalog_nnz:
            assert catalog_nnz[mid] == features_nnz[mid], (
                f"nnz mismatch for {mid}: catalog={catalog_nnz[mid]}, "
                f"features={features_nnz[mid]}"
            )

    def test_feature_dimensions_positive(self, tmp_metadata):
        with open(tmp_metadata["features"], "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                assert int(row["rows"]) > 0
                assert int(row["cols"]) > 0

    def test_max_row_nnz_ge_min(self, tmp_metadata):
        with open(tmp_metadata["features"], "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                assert int(row["max_row_nnz"]) >= int(row["min_row_nnz"])

    def test_mean_consistent_with_nnz_rows(self, tmp_metadata):
        """mean_row_nnz must equal nnz / rows within 1e-5."""
        with open(tmp_metadata["features"], "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                expected = int(row["nnz"]) / int(row["rows"])
                actual   = float(row["mean_row_nnz"])
                assert abs(actual - expected) < 1e-5, (
                    f"mean_row_nnz mismatch for {row['matrix_id']}: "
                    f"expected {expected:.6f}, got {actual:.6f}"
                )


# =========================================================================== #
# validate_csv_headers utility                                                  #
# =========================================================================== #

class TestValidateCsvHeaders:
    def test_correct_headers_returns_true(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b,c\n1,2,3\n")
        assert validate_csv_headers(p, ["a", "b", "c"]) is True

    def test_wrong_headers_returns_false(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("x,y\n1,2\n")
        assert validate_csv_headers(p, ["a", "b"]) is False

    def test_missing_file_returns_false(self, tmp_path):
        assert validate_csv_headers(tmp_path / "nonexistent.csv", ["a"]) is False

    def test_empty_file_returns_false(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        assert validate_csv_headers(p, ["a"]) is False
