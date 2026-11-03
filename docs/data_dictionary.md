# Data Dictionary & Shared CSV Contracts

This document specifies the exact schema contracts for shared metadata and benchmark output files. These contracts decouple data ingestion, GPU execution, and model evaluation across team members.

---

## 1. Matrix Catalog (`data/metadata/matrix_catalog.csv`)

**Owner / Producer:** Member 2  
**Consumers:** Member 1, Member 3  
**Description:** Inventory of all sparse matrices included in the experimental evaluation.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `matrix_id` | string | Unique identifier for the matrix | `web-Google` |
| `rows` | integer | Number of rows ($M$) | `916428` |
| `cols` | integer | Number of columns ($N$) | `916428` |
| `nnz` | integer | Total count of non-zero entries | `5105039` |
| `matrix_group` | string | Origin domain or category (e.g., SuiteSparse class, synthetic) | `directed_graph` |

**Header row:**
```csv
matrix_id,rows,cols,nnz,matrix_group
```

---

## 2. Matrix Features (`data/metadata/features.csv`)

**Owner / Producer:** Member 2  
**Consumers:** Member 3  
**Description:** Extracted structural features for each matrix used as input to the kernel selector.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `matrix_id` | string | Unique identifier matching `matrix_catalog.csv` | `web-Google` |
| `rows` | integer | Number of rows ($M$) | `916428` |
| `cols` | integer | Number of columns ($N$) | `916428` |
| `nnz` | integer | Total non-zero elements | `5105039` |
| `mean_row_nnz` | float | Average non-zeros per row ($\text{nnz} / \text{rows}$) | `5.57` |
| `variance_row_nnz` | float | Variance in row non-zero counts | `142.38` |
| `max_row_nnz` | integer | Maximum non-zeros in any single row | `6326` |
| `min_row_nnz` | integer | Minimum non-zeros in any single row | `0` |

**Header row:**
```csv
matrix_id,rows,cols,nnz,mean_row_nnz,variance_row_nnz,max_row_nnz,min_row_nnz
```

---

## 3. Dataset Splits (`data/metadata/splits.csv`)

**Owner / Producer:** Member 2  
**Consumers:** Member 3  
**Description:** Partitioning of matrices for train, validation, and out-of-distribution / held-out test evaluation.

| Column Name | Data Type | Description | Allowed Values |
| :--- | :--- | :--- | :--- |
| `matrix_id` | string | Unique identifier matching `matrix_catalog.csv` | Any valid `matrix_id` |
| `split` | string | Dataset partition assignment | `train`, `val`, `test` |

**Header row:**
```csv
matrix_id,split
```

---

## 4. Kernel Timings (`benchmarks/kernel_timings.csv`)

**Owner / Producer:** Member 1 (CUDA machine)  
**Consumers:** Member 3  
**Description:** Measured runtime and verification status for each evaluated kernel on each matrix.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `matrix_id` | string | Unique identifier matching `matrix_catalog.csv` | `web-Google` |
| `kernel_name` | string | Identifier of the CUDA SpMV kernel | `csr_vector` |
| `kernel_time_ms` | float | Steady-state kernel execution time in milliseconds | `1.42` |
| `preprocessing_time_ms` | float | Time spent in format conversion or per-matrix prep (0 if none) | `0.0` |
| `correctness_pass` | boolean | Validation check against CPU reference (`True`/`False`) | `True` |

**Header row:**
```csv
matrix_id,kernel_name,kernel_time_ms,preprocessing_time_ms,correctness_pass
```

