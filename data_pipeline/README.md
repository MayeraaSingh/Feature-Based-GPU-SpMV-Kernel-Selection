# Data Pipeline Module

**Owner:** Member 2 (Dataset curation, CSR data pipeline, CPU reference, and feature extraction)

## Purpose

The `data_pipeline` module is responsible for:
1. Downloading and ingesting raw matrix datasets (e.g., Matrix Market format).
2. Validating matrix structure and converting to canonical Compressed Sparse Row (CSR) representation (`row_ptr`, `col_idx`, `values`).
3. Providing a deterministic CPU reference SpMV implementation for verification.
4. Extracting structural features (e.g., density, row non-zero statistics) into `data/metadata/features.csv`.
5. Generating train/validation/test splits into `data/metadata/splits.csv`.

