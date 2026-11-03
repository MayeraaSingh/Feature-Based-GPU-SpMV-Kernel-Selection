"""
test_reference.py
=================
Tests for the CPU reference SpMV implementation (reference_spmv.py).

Validates:
  - Correct computation of y = A x
  - Reproducibility of the reference vector
  - check_result tolerance logic
  - Edge cases (empty matrix, 1×1 matrix, rectangular matrix)

Run with:
    pytest tests/test_reference.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from data_pipeline.reference_spmv import cpu_spmv, reference_vector, check_result


class TestReferenceVector:
    def test_shape(self):
        x = reference_vector(100)
        assert x.shape == (100,)

    def test_values_in_unit_interval(self):
        x = reference_vector(100, seed=0)
        assert np.all(x >= 0.0)
        assert np.all(x < 1.0)

    def test_reproducibility(self):
        x1 = reference_vector(50, seed=42)
        x2 = reference_vector(50, seed=42)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_differ(self):
        x1 = reference_vector(50, seed=1)
        x2 = reference_vector(50, seed=2)
        assert not np.allclose(x1, x2)

    def test_dtype_float64(self):
        x = reference_vector(10, dtype=np.float64)
        assert x.dtype == np.float64

    def test_dtype_float32(self):
        x = reference_vector(10, dtype=np.float32)
        assert x.dtype == np.float32


class TestCpuSpmv:
    def test_diagonal_identity(self):
        """I * x == x for any x."""
        n = 20
        mat = sp.eye(n, format="csr")
        x = reference_vector(n, seed=7)
        y, x_used = cpu_spmv(mat, x=x)
        np.testing.assert_allclose(y, x, rtol=1e-12)

    def test_known_result(self):
        """Hand-crafted 2×2 case: [[2,0],[0,3]] * [1, 2] = [2, 6]."""
        mat = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
        x = np.array([1.0, 2.0])
        y, _ = cpu_spmv(mat, x=x)
        np.testing.assert_allclose(y, [2.0, 6.0], rtol=1e-12)

    def test_auto_x_generation(self):
        """When x is None, x is generated reproducibly from the seed."""
        mat = sp.eye(10, format="csr")
        y1, x1 = cpu_spmv(mat, seed=99)
        y2, x2 = cpu_spmv(mat, seed=99)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_rectangular_matrix(self):
        """Rectangular matrix: rows ≠ cols."""
        data = np.ones(5)
        row = np.arange(5)
        col = np.arange(5)
        mat = sp.csr_matrix((data, (row, col)), shape=(5, 10))
        x = reference_vector(10, seed=3)
        y, _ = cpu_spmv(mat, x=x)
        assert y.shape == (5,)

    def test_empty_matrix(self):
        """All-zero matrix: y must be all zeros."""
        mat = sp.csr_matrix((5, 5))
        y, _ = cpu_spmv(mat)
        np.testing.assert_array_equal(y, np.zeros(5))

    def test_wrong_x_shape_raises(self):
        mat = sp.eye(5, format="csr")
        bad_x = np.ones(3)
        with pytest.raises(ValueError, match="shape"):
            cpu_spmv(mat, x=bad_x)

    def test_non_sparse_raises(self):
        dense = np.eye(5)
        with pytest.raises(TypeError, match="SciPy sparse"):
            cpu_spmv(dense)

    def test_output_dtype_float64(self):
        """Output is always float64 regardless of matrix dtype."""
        mat = sp.eye(5, format="csr", dtype=np.float32)
        y, _ = cpu_spmv(mat)
        assert y.dtype == np.float64

    def test_matches_scipy_dot(self):
        """Result must match scipy dense dot product to machine precision."""
        import numpy.random as npr
        rng = np.random.default_rng(42)
        dense = rng.random((8, 8))
        dense[dense < 0.5] = 0.0
        mat = sp.csr_matrix(dense)
        x = reference_vector(8, seed=5)
        y_ref_scipy = mat.toarray() @ x
        y, _ = cpu_spmv(mat, x=x)
        np.testing.assert_allclose(y, y_ref_scipy, rtol=1e-12)


class TestCheckResult:
    def _setup(self, n=10):
        mat = sp.eye(n, format="csr", dtype=np.float64)
        x = reference_vector(n, seed=0)
        y_ref, _ = cpu_spmv(mat, x=x)
        return mat, x, y_ref

    def test_exact_match_passes(self):
        mat, x, y_ref = self._setup()
        passed, err, thresh = check_result(y_ref, y_ref, mat, x)
        assert passed is True
        assert err == pytest.approx(0.0, abs=1e-16)

    def test_within_tolerance_passes(self):
        mat, x, y_ref = self._setup()
        # Small perturbation well within tolerance
        y_test = y_ref + 1e-9
        passed, err, thresh = check_result(y_ref, y_test, mat, x, tolerance=1e-5)
        assert passed is True

    def test_outside_tolerance_fails(self):
        mat, x, y_ref = self._setup()
        y_bad = y_ref + 1.0  # large error
        passed, err, thresh = check_result(y_ref, y_bad, mat, x, tolerance=1e-5)
        assert passed is False
        assert err > thresh

    def test_shape_mismatch_raises(self):
        mat = sp.eye(5, format="csr")
        x = reference_vector(5, seed=0)
        y_ref = np.ones(5)
        y_wrong = np.ones(4)
        with pytest.raises(ValueError, match="Shape mismatch"):
            check_result(y_ref, y_wrong, mat, x)

    def test_zero_matrix_passes(self):
        """For an all-zero matrix, y_ref == y_test == 0 always passes."""
        mat = sp.csr_matrix((5, 5))
        x = reference_vector(5, seed=0)
        y_ref = np.zeros(5)
        passed, err, thresh = check_result(y_ref, y_ref, mat, x)
        assert passed is True
