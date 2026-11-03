# Tests Module

**Owner:** Shared / Collaborative

## Purpose

The `tests` module contains automated unit tests and integration tests to ensure code quality and contract compliance across components:

- `test_data_pipeline`: Validates CSR conversion, non-zero handling, and feature extraction accuracy.
- `test_reference`: Validates CPU reference SpMV outputs against standard numerical libraries (`scipy.sparse`).
- `test_selector`: Tests training routines, inference pipeline, and model persistence.
- `test_contracts`: Validates that generated metadata CSV files conform to schemas specified in `docs/data_dictionary.md`.

