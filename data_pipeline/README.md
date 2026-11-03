# Data Pipeline — Member 2

**Owner:** Member 2  
**Primary directories:** `data_pipeline/`, `data/`, `configs/`, `tests/`

---

## Purpose

This module implements the full Member 2 data pipeline for the SpMV kernel selection project:

1. **Matrix loading** — load `.mtx` (Matrix Market) and `.npz` (SciPy sparse) files from a local directory.
2. **CSR validation** — validate and canonicalize each matrix into Compressed Sparse Row format.
3. **Feature extraction** — compute the seven agreed structural features.
4. **Split generation** — produce a reproducible train / val / test manifest.
5. **CSV output** — write the three shared metadata CSV files consumed by Members 1 and 3.
6. **CPU reference SpMV** — provide a trusted `y = Ax` implementation for GPU correctness checks.

---

## Module structure

```
data_pipeline/
  __init__.py          Public API re-exports
  loader.py            Matrix file loading and stable matrix_id assignment
  validator.py         CSR structural validation and canonicalization
  features.py          Structural feature extraction
  reference_spmv.py    CPU reference SpMV (y = A x) and correctness checker
  splits.py            Train / val / test split manifest generation
  catalog.py           Write metadata CSVs; CSV header validation utility
  pipeline.py          Top-level orchestration
  pilot_data.py        Synthetic pilot matrix generator (dev/test only)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with synthetic pilot data (development / CI)

```bash
python scripts/run_pipeline.py --pilot
```

This generates seven synthetic sparse matrices in `data/raw/`, processes them, and writes:

- `data/metadata/matrix_catalog.csv`
- `data/metadata/features.csv`
- `data/metadata/splits.csv`
- `data/metadata/exclusion_log.csv`

### 3. Run with real matrices

Place your `.mtx` or `.npz` sparse matrix files in `data/raw/` (or configure a different directory in `configs/dataset_config.yaml`), then:

```bash
python scripts/run_pipeline.py
```

To override the raw directory:

```bash
python scripts/run_pipeline.py --raw-dir /path/to/matrices/
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Matrix source format

| Format | Extension | Notes |
|---|---|---|
| Matrix Market | `.mtx` | Loaded via `scipy.io.mmread` |
| SciPy sparse | `.npz` | Saved with `scipy.sparse.save_npz` |
| Raw CSR archive | `.npz` | NumPy archive with keys `data`, `indices`, `indptr`, `shape` |

SuiteSparse downloads are **not automated** by this pipeline.  Place pre-downloaded files in `data/raw/` and run the pipeline normally.  The pipeline is dataset-source agnostic.

---

## Stable matrix_id convention

A matrix's `matrix_id` is derived from the **filename stem** of its source file, with non-alphanumeric characters replaced by underscores and leading/trailing underscores stripped.  For example:

| Filename | matrix_id |
|---|---|
| `cage4.mtx` | `cage4` |
| `web-Google.mtx` | `web-Google` |
| `my matrix.npz` | `my_matrix` |

IDs are **not** based on file ordering or directory listing position, making them stable across pipeline runs and machines.  If two filenames produce the same ID (collision), a numeric suffix is appended and a warning is logged.  Collisions should be resolved by renaming source files.

---

## CSR validation policy

| Issue | Policy |
|---|---|
| Duplicate (i, j) entries | Summed into a single entry; warning logged |
| Explicit zeros (after summation) | Removed; warning logged |
| NaN or Inf values | Hard error (ValidationError) |
| Negative column indices | Hard error |
| Column index ≥ cols | Hard error |
| Non-monotonic row pointers | Hard error |
| Empty rows (zero nnz) | Allowed; counted in all statistics |
| Index dtype | int32 for matrices ≤ 2³¹−1 dimensions; int64 otherwise |

---

## Feature definitions

All features are extracted from the **CSR row-pointer array** (`indptr`).

| Feature | Definition |
|---|---|
| `rows` | Number of matrix rows |
| `cols` | Number of matrix columns |
| `nnz` | Number of nonzero entries after canonicalization |
| `mean_row_nnz` | `nnz / rows` |
| `variance_row_nnz` | Population variance of per-row nnz counts (`ddof=0`) |
| `max_row_nnz` | Maximum per-row nnz count |
| `min_row_nnz` | Minimum per-row nnz count (0 if any empty row exists) |

**Variance convention:** Population variance (`ddof=0`, NumPy default) is used consistently for all matrices.  Sample variance (`ddof=1`) is not used.

**Empty rows:** Rows with zero nonzeros are valid inputs.  They are included in all statistical calculations (contributing 0 to the mean and potentially increasing variance).

---

## Split strategy

The default strategy is **group-stratified** (`strategy="group"`):

- All matrices in a `matrix_group` are assigned to the **same split** (train, val, or test).
- Groups are shuffled randomly with a fixed seed and then packed into splits to approximately match the target proportions.
- This prevents structurally similar matrices (e.g., all finite-element meshes) from appearing in both train and test splits.

A `strategy="random"` option ignores groups and randomly assigns individual matrices.  It is available for comparison but the default group strategy is preferred for the research.

**Configuration** (`configs/dataset_config.yaml`):

```yaml
splits:
  train_ratio: 0.70
  val_ratio:   0.15
  test_ratio:  0.15
  random_seed: 42
  stratify_by: matrix_group   # use "matrix_group" for group strategy
```

The split manifest must be kept fixed for a reported experiment.  Do not revise splits after seeing test outcomes.

---

## CPU reference SpMV

`data_pipeline/reference_spmv.py` provides:

- `reference_vector(cols, seed)` — generate a reproducible input vector `x`.
- `cpu_spmv(csr, x, seed)` — compute `y = A x` in float64 using SciPy.
- `check_result(y_ref, y_test, A, x, tolerance)` — validate GPU output against the reference using the benchmark protocol criterion:

```
‖y_gpu − y_cpu‖_∞  ≤  ε · (‖A‖_∞ · ‖x‖_∞ + 1)
```

Default `ε = 1e-5` (appropriate for float32 GPU results).  Use `ε = 1e-10` for float64.

Member 1 should call `cpu_spmv` with the same matrix and vector (`seed=42` by default) that the GPU kernel uses.

---

## Output file contracts

### `data/metadata/matrix_catalog.csv`
```
matrix_id,rows,cols,nnz,matrix_group
```

### `data/metadata/features.csv`
```
matrix_id,rows,cols,nnz,mean_row_nnz,variance_row_nnz,max_row_nnz,min_row_nnz
```

### `data/metadata/splits.csv`
```
matrix_id,split
```
Split values: `train`, `val`, `test`.

### `data/metadata/exclusion_log.csv`
```
filename,reason
```
Records any matrix files that could not be loaded or validated.

---

## Pilot dataset (development only)

`data_pipeline/pilot_data.py` generates seven clearly labeled synthetic sparse matrices spanning three structural groups:

| matrix_id | Description | Group |
|---|---|---|
| `pilot_diag_100` | 100×100 diagonal | `synthetic_diagonal` |
| `pilot_dense_block_50` | 50×50 fully dense | `synthetic_dense` |
| `pilot_uniform_200` | 200×200, uniform 10 nnz/row | `synthetic_uniform` |
| `pilot_skew_150` | 150×150, heavy-tail row lengths | `synthetic_skewed` |
| `pilot_banded_80` | 80×80, bandwidth-5 banded | `synthetic_banded` |
| `pilot_empty_rows_60` | 60×60, ~30% empty rows | `synthetic_sparse` |
| `pilot_large_sparse` | 500×500, ~3 nnz/row avg | `synthetic_sparse` |

> **These are synthetic matrices for pipeline development and testing only.**  
> They must never be used as the final research dataset or as final benchmark evidence.

---

## What Members 1 and 3 receive

**Member 1 (GPU benchmarking):**
- `data/metadata/matrix_catalog.csv` — list of matrix IDs to benchmark.
- Matrix files in `data/raw/` — same files the pipeline loaded, for GPU kernel input.
- `cpu_spmv` / `reference_vector` from `data_pipeline.reference_spmv` — to compute correctness reference `y = Ax` with the same input vector.

**Member 3 (selector / evaluation):**
- `data/metadata/features.csv` — feature input for selector training.
- `data/metadata/splits.csv` — leakage-controlled train/val/test assignments.

All three files use the same stable `matrix_id`.

---

## Unresolved decisions (team agreement required before final run)

1. **Final dataset source** — SuiteSparse is currently unavailable for download.  The team must agree on the final matrix collection and provide files in `data/raw/`.
2. **Final split strategy** — the group-based strategy is implemented; the team (Members 2 and 3) should confirm this is acceptable before finalizing the split manifest for the reported experiment.
3. **Matrix group labels** — for real datasets, group labels (e.g., `directed_graph`, `finite_element`) must be agreed and supplied via `configs/dataset_config.yaml` or a supplementary metadata file.
4. **Numerical tolerance** — Member 1 and Member 2 should jointly confirm the `ε` value used in `check_result` for their GPU dtype (float32 vs float64).
