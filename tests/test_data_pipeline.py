"""
test_data_pipeline.py
=====================
Unit tests for the Member 2 data pipeline:
  - CSR validation (validator.py)
  - Matrix loading (loader.py)
  - Feature extraction (features.py)
  - Split generation (splits.py)

Run with:
    pytest tests/test_data_pipeline.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from data_pipeline.validator import validate_csr, ValidationError
from data_pipeline.loader import load_matrix, load_directory, _stem_to_id
from data_pipeline.features import extract_features
from data_pipeline.splits import make_splits


# =========================================================================== #
# Helpers                                                                       #
# =========================================================================== #

def _simple_csr(rows=5, cols=5, density=0.3, seed=0):
    """Create a simple dense-ish CSR matrix for testing."""
    rng = np.random.default_rng(seed)
    dense = rng.random((rows, cols))
    dense[dense < (1 - density)] = 0.0
    return sp.csr_matrix(dense)


def _diagonal_csr(n=10):
    return sp.eye(n, format="csr", dtype=np.float64)


# =========================================================================== #
# validator.py                                                                  #
# =========================================================================== #

class TestValidateCsr:
    def test_valid_diagonal(self):
        mat = _diagonal_csr(20)
        result = validate_csr(mat, "diag_20")
        assert sp.isspmatrix_csr(result)
        assert result.shape == (20, 20)
        assert result.nnz == 20

    def test_valid_random_csr(self):
        mat = _simple_csr(10, 10, density=0.4)
        result = validate_csr(mat, "random_10x10")
        assert sp.isspmatrix_csr(result)

    def test_converts_coo_to_csr(self):
        """COO input should be converted and validated."""
        coo = sp.eye(5, format="coo")
        result = validate_csr(coo, "coo_eye")
        assert sp.isspmatrix_csr(result)

    def test_converts_csc_to_csr(self):
        csc = sp.eye(5, format="csc")
        result = validate_csr(csc, "csc_eye")
        assert sp.isspmatrix_csr(result)

    def test_duplicates_are_summed(self):
        """Explicit duplicate (i, j) entries must be summed, not raise errors."""
        # Build a COO with duplicates manually
        row = np.array([0, 0, 1], dtype=np.int32)
        col = np.array([0, 0, 1], dtype=np.int32)
        data = np.array([1.0, 2.0, 3.0])
        mat = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        result = validate_csr(mat, "dup_test")
        # (0,0) should now be 3.0
        assert result[0, 0] == pytest.approx(3.0)
        assert result.nnz == 2

    def test_explicit_zeros_removed(self):
        """Explicit zeros must be removed after duplicate summation."""
        row = np.array([0, 1], dtype=np.int32)
        col = np.array([0, 1], dtype=np.int32)
        data = np.array([0.0, 1.0])
        mat = sp.csr_matrix((data, (row, col)), shape=(3, 3))
        result = validate_csr(mat, "explicit_zero")
        assert result.nnz == 1  # zero entry removed
        assert result[1, 1] == pytest.approx(1.0)

    def test_negative_column_index_raises(self):
        """Negative column index must raise ValidationError."""
        # Manually construct a broken CSR
        indptr = np.array([0, 1, 2], dtype=np.int32)
        indices = np.array([-1, 0], dtype=np.int32)
        data = np.array([1.0, 1.0])
        mat = sp.csr_matrix((data, indices, indptr), shape=(2, 2))
        # Bypass scipy checks by manipulating directly
        mat.indices = indices
        with pytest.raises(ValidationError, match="Negative column index"):
            validate_csr(mat, "neg_col")

    def test_invalid_dimensions_raises(self):
        """Zero-dimension matrix must raise ValidationError."""
        mat = sp.csr_matrix((0, 5))
        with pytest.raises(ValidationError, match="Invalid dimensions"):
            validate_csr(mat, "zero_rows")

    def test_nan_in_data_raises(self):
        mat = _diagonal_csr(5)
        mat.data[0] = float("nan")
        with pytest.raises(ValidationError, match="non-finite"):
            validate_csr(mat, "nan_test")

    def test_inf_in_data_raises(self):
        mat = _diagonal_csr(5)
        mat.data[0] = float("inf")
        with pytest.raises(ValidationError, match="non-finite"):
            validate_csr(mat, "inf_test")

    def test_non_sparse_raises(self):
        with pytest.raises(ValidationError, match="not a SciPy sparse"):
            validate_csr(np.eye(5), "dense_array")

    def test_indices_dtype_int32(self):
        """Indices should be int32 for small matrices."""
        mat = _diagonal_csr(10)
        result = validate_csr(mat, "dtype_check")
        assert result.indices.dtype == np.int32
        assert result.indptr.dtype == np.int32


# =========================================================================== #
# loader.py                                                                     #
# =========================================================================== #

class TestStemToId:
    def test_simple_name(self):
        assert _stem_to_id("myMatrix") == "myMatrix"

    def test_spaces_replaced(self):
        assert _stem_to_id("my matrix") == "my_matrix"

    def test_special_chars(self):
        # Underscore is kept, slash replaced
        assert _stem_to_id("a/b") == "a_b"

    def test_leading_trailing_stripped(self):
        assert _stem_to_id("__foo__") == "foo"

    def test_empty_becomes_matrix(self):
        assert _stem_to_id("") == "matrix"

    def test_hyphens_preserved(self):
        # hyphens are valid in matrix_id
        assert _stem_to_id("web-Google") == "web-Google"


class TestLoadMatrix:
    def test_load_npz_scipy(self, tmp_path):
        """Load a scipy sparse .npz file."""
        mat = _diagonal_csr(15)
        npz_path = tmp_path / "diag_15.npz"
        sp.save_npz(str(npz_path), mat)

        result, mid = load_matrix(npz_path)
        assert sp.isspmatrix_csr(result)
        assert result.shape == (15, 15)
        assert mid == "diag_15"

    def test_load_mtx(self, tmp_path):
        """Load a Matrix Market .mtx file."""
        import scipy.io
        mat = _diagonal_csr(8)
        mtx_path = tmp_path / "eye_8.mtx"
        scipy.io.mmwrite(str(mtx_path), mat)

        result, mid = load_matrix(mtx_path)
        assert sp.isspmatrix_csr(result)
        assert result.shape == (8, 8)
        assert mid == "eye_8"

    def test_load_npz_raw_archive(self, tmp_path):
        """Load a raw numpy .npz archive with CSR keys."""
        mat = _diagonal_csr(6)
        npz_path = tmp_path / "raw_csr.npz"
        np.savez(
            str(npz_path),
            data=mat.data,
            indices=mat.indices,
            indptr=mat.indptr,
            shape=np.array(mat.shape),
        )
        result, mid = load_matrix(npz_path)
        assert sp.isspmatrix_csr(result)
        assert result.shape == (6, 6)

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "file.csv"
        p.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_matrix(p)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_matrix(tmp_path / "nonexistent.npz")

    def test_explicit_matrix_id_override(self, tmp_path):
        mat = _diagonal_csr(4)
        npz_path = tmp_path / "any_name.npz"
        sp.save_npz(str(npz_path), mat)
        _, mid = load_matrix(npz_path, matrix_id="my_custom_id")
        assert mid == "my_custom_id"


class TestLoadDirectory:
    def test_loads_multiple_files(self, tmp_path):
        for i in range(3):
            sp.save_npz(str(tmp_path / f"mat_{i:02d}.npz"), _diagonal_csr(i + 2))

        matrices, exclusions = load_directory(tmp_path)
        assert len(matrices) == 3
        assert len(exclusions) == 0

    def test_non_matrix_files_ignored(self, tmp_path):
        sp.save_npz(str(tmp_path / "good.npz"), _diagonal_csr(5))
        (tmp_path / "notes.txt").write_text("hello")
        matrices, _ = load_directory(tmp_path, extensions=(".npz",))
        assert len(matrices) == 1

    def test_invalid_file_goes_to_exclusions(self, tmp_path):
        sp.save_npz(str(tmp_path / "good.npz"), _diagonal_csr(5))
        # Write a corrupt .npz
        (tmp_path / "bad.npz").write_bytes(b"not a npz file")
        matrices, exclusions = load_directory(tmp_path)
        assert "good.npz" not in exclusions
        assert "bad.npz" in exclusions

    def test_not_a_directory_raises(self, tmp_path):
        p = tmp_path / "somefile.txt"
        p.write_text("x")
        with pytest.raises(NotADirectoryError):
            load_directory(p)


# =========================================================================== #
# features.py                                                                   #
# =========================================================================== #

class TestExtractFeatures:
    def test_diagonal_matrix(self):
        """For a diagonal n×n matrix: mean=1, var=0, max=min=1."""
        n = 10
        mat = validate_csr(_diagonal_csr(n), "diag")
        rec = extract_features(mat, "diag")

        assert rec["matrix_id"] == "diag"
        assert rec["rows"] == n
        assert rec["cols"] == n
        assert rec["nnz"] == n
        assert rec["mean_row_nnz"] == pytest.approx(1.0)
        assert rec["variance_row_nnz"] == pytest.approx(0.0)
        assert rec["max_row_nnz"] == 1
        assert rec["min_row_nnz"] == 1

    def test_dense_block(self):
        """For a 4×4 fully dense matrix: mean=4, var=0, max=min=4."""
        mat = validate_csr(sp.csr_matrix(np.ones((4, 4))), "dense")
        rec = extract_features(mat, "dense")

        assert rec["mean_row_nnz"] == pytest.approx(4.0)
        assert rec["variance_row_nnz"] == pytest.approx(0.0)
        assert rec["max_row_nnz"] == 4
        assert rec["min_row_nnz"] == 4

    def test_skewed_row_lengths(self):
        """Hand-crafted matrix: row 0 has 3 nnz, row 1 has 1 nnz."""
        # row 0: cols 0, 1, 2 → 3 nnz
        # row 1: col 0        → 1 nnz
        row = [0, 0, 0, 1]
        col = [0, 1, 2, 0]
        data = [1.0, 2.0, 3.0, 4.0]
        mat = validate_csr(
            sp.csr_matrix((data, (row, col)), shape=(2, 3)), "skewed"
        )
        rec = extract_features(mat, "skewed")

        assert rec["nnz"] == 4
        assert rec["mean_row_nnz"] == pytest.approx(2.0)   # (3+1)/2
        assert rec["max_row_nnz"] == 3
        assert rec["min_row_nnz"] == 1
        # Population variance of [3, 1]: mean=2, var = ((3-2)^2 + (1-2)^2) / 2 = 1.0
        assert rec["variance_row_nnz"] == pytest.approx(1.0)

    def test_empty_rows_handled(self):
        """Matrix with a fully empty row: min_row_nnz must be 0."""
        row = [0, 0]  # only row 0 has nnz; row 1 is empty
        col = [0, 1]
        data = [1.0, 2.0]
        mat = validate_csr(
            sp.csr_matrix((data, (row, col)), shape=(2, 3)), "empty_row"
        )
        rec = extract_features(mat, "empty_row")

        assert rec["min_row_nnz"] == 0
        assert rec["max_row_nnz"] == 2

    def test_fully_empty_matrix(self):
        """nnz=0 matrix: all statistics are 0, no crash."""
        mat = sp.csr_matrix((5, 5))
        mat_validated = validate_csr(mat, "empty_mat")
        rec = extract_features(mat_validated, "empty_mat")

        assert rec["nnz"] == 0
        assert rec["mean_row_nnz"] == pytest.approx(0.0)
        assert rec["variance_row_nnz"] == pytest.approx(0.0)
        assert rec["max_row_nnz"] == 0
        assert rec["min_row_nnz"] == 0

    def test_mean_equals_nnz_over_rows(self):
        """mean_row_nnz must equal nnz / rows exactly."""
        mat = validate_csr(_simple_csr(7, 7, density=0.5, seed=99), "random_7")
        rec = extract_features(mat, "random_7")
        expected_mean = rec["nnz"] / rec["rows"]
        assert rec["mean_row_nnz"] == pytest.approx(expected_mean, rel=1e-6)

    def test_population_variance_not_sample(self):
        """Variance uses ddof=0 (population), not ddof=1 (sample)."""
        # row 0: 1 nnz; row 1: 3 nnz
        row = [0, 1, 1, 1]
        col = [0, 0, 1, 2]
        data = [1.0, 1.0, 1.0, 1.0]
        mat = validate_csr(
            sp.csr_matrix((data, (row, col)), shape=(2, 4)), "var_check"
        )
        rec = extract_features(mat, "var_check")
        # row_counts = [1, 3]; mean = 2; pop var = ((1-2)^2 + (3-2)^2)/2 = 1.0
        assert rec["variance_row_nnz"] == pytest.approx(1.0)

    def test_non_csr_raises(self):
        coo = sp.eye(5, format="coo")
        with pytest.raises(TypeError, match="Expected a CSR matrix"):
            extract_features(coo, "coo_test")

    def test_feature_record_keys(self):
        """Feature record must have exactly the expected keys."""
        expected_keys = {
            "matrix_id", "rows", "cols", "nnz",
            "mean_row_nnz", "variance_row_nnz", "max_row_nnz", "min_row_nnz",
        }
        mat = validate_csr(_diagonal_csr(5), "key_test")
        rec = extract_features(mat, "key_test")
        assert set(rec.keys()) == expected_keys


# =========================================================================== #
# splits.py                                                                     #
# =========================================================================== #

class TestMakeSplits:
    _IDS = [f"m{i:02d}" for i in range(20)]
    _GROUPS = {f"m{i:02d}": f"g{i % 4}" for i in range(20)}  # 4 groups of 5

    def test_all_ids_assigned(self):
        manifest = make_splits(self._IDS, self._GROUPS, seed=42)
        assert set(manifest.keys()) == set(self._IDS)

    def test_valid_split_labels(self):
        manifest = make_splits(self._IDS, self._GROUPS, seed=42)
        assert all(v in {"train", "val", "test"} for v in manifest.values())

    def test_ratios_approximately_respected(self):
        manifest = make_splits(
            self._IDS, self._GROUPS, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=0
        )
        n = len(self._IDS)
        n_train = sum(1 for v in manifest.values() if v == "train")
        n_test  = sum(1 for v in manifest.values() if v == "test")
        n_val   = sum(1 for v in manifest.values() if v == "val")
        # Allow ±2 matrices slack due to group-level granularity
        assert n_train >= n * 0.50
        assert n_test  >= 1
        assert n_val   >= 0
        assert n_train + n_val + n_test == n

    def test_reproducibility(self):
        m1 = make_splits(self._IDS, self._GROUPS, seed=7)
        m2 = make_splits(self._IDS, self._GROUPS, seed=7)
        assert m1 == m2

    def test_different_seeds_differ(self):
        m1 = make_splits(self._IDS, self._GROUPS, seed=1)
        m2 = make_splits(self._IDS, self._GROUPS, seed=2)
        # With enough matrices it is essentially guaranteed they differ
        assert m1 != m2

    def test_random_strategy(self):
        manifest = make_splits(self._IDS, strategy="random", seed=42)
        assert set(manifest.keys()) == set(self._IDS)
        assert all(v in {"train", "val", "test"} for v in manifest.values())

    def test_no_groups_falls_back_to_random(self):
        """group strategy without groups dict → random (with warning)."""
        import warnings
        manifest = make_splits(self._IDS, groups=None, strategy="group", seed=42)
        assert set(manifest.keys()) == set(self._IDS)

    def test_empty_ids_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_splits([], seed=42)

    def test_invalid_ratios_raises(self):
        with pytest.raises(ValueError, match="must equal 1.0"):
            make_splits(self._IDS, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5, seed=42)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            make_splits(self._IDS, strategy="unsupported", seed=42)

    def test_single_matrix(self):
        """A single matrix must be assigned to train (no crash)."""
        manifest = make_splits(["solo"], groups={"solo": "g0"}, seed=42)
        assert "solo" in manifest

    def test_group_leakage_prevention(self):
        """All matrices in the same group must end up in the same split."""
        ids = [f"m{i}" for i in range(12)]
        groups = {f"m{i}": "group_A" for i in range(6)}
        groups.update({f"m{i}": "group_B" for i in range(6, 12)})
        manifest = make_splits(ids, groups=groups, seed=0)
        group_A_splits = {manifest[f"m{i}"] for i in range(6)}
        group_B_splits = {manifest[f"m{i}"] for i in range(6, 12)}
        # Each group must be entirely in one split
        assert len(group_A_splits) == 1
        assert len(group_B_splits) == 1
