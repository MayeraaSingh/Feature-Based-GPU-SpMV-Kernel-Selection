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


---

## Additions for the real GPU experiment (contracts above are unchanged)

All files below are additional; none replaces a contract file.

### `data/metadata/real/` (real-data copies of the shared contracts, produced with `configs/dataset_real.yaml`)
Same schemas as `matrix_catalog.csv`, `features.csv`, `splits.csv`, plus:
* `features_extended.csv`: `matrix_id,cv_row_nnz,empty_row_frac,max_over_mean,frac_rows_gt_32` (derived from the row-pointer array only; see `data_pipeline/features_extended.py`).
* `redundancy_log.csv`: `matrix_id,kept_representative,reason,pattern_sha256` (exact duplicate sparsity patterns that were dropped).

### `data/metadata/dataset_manifest.csv` (provenance)
`matrix_id,name,source,source_url,license,collection,set,matrix_group,application_domain,rows,cols,nnz_stored,nnz_csr,sparsity,format,field,symmetry,acquired_date,sha256_gz,sha256_mtx,download_status,validation_status,note`.

### Benchmark directory `results/<run_id>/benchmark/`
| File | Columns / content |
|---|---|
| `kernel_timings.csv` | the unchanged contract: `matrix_id,kernel_name,kernel_time_ms,preprocessing_time_ms,correctness_pass`. `kernel_time_ms` = median over 100 repetitions of the per-SpMV time (each repetition averages a block of `inner_iters` back-to-back SpMVs); `preprocessing_time_ms` = median host wall-clock of `prepare()+bind()` |
| `raw_measurements.csv` | `matrix_id,phase,kernel,rep,ms` (`phase` = `kernel` or `prep`; kernel values are per-SpMV) |
| `benchmark_summary.csv` | per matrix and kernel: `n_reps,median_ms,mean_ms,std_ms,iqr_ms,min_ms,max_ms,cv,n_outliers,prep_median_ms,prep_std_ms,timing_mode,inner_iters,max_abs_err,threshold,err_over_threshold,correct_first,correct_last` |
| `overhead_measurements.csv` | `matrix_id,alloc_ms,h2d_ms,feature_cpu_ms,feature_gpu_ms,feature_gpu_matches_cpu,timer_floor_ms,mem_needed_mb_est` |
| `host_stage_timings.csv` | `matrix_id,load_validate_ms,feature_extract_python_ms,export_ms` (Python wall clock) |
| `gpu_telemetry.csv` | `nvidia-smi` samples (temperature, SM and memory clock, utilisation, memory, power, P-state, throttle reasons) before and after each matrix |
| `benchmark_failures.csv` | `matrix_id,kernel_name,stage,reason` (kernels that did not run to completion; no row exists for them in `kernel_timings.csv`) |
| `environment.json`, `benchmark_manifest.json` | GPU / driver / CUDA / compiler / package versions, git state, protocol parameters |

### Experiment run directory `results/<run_id>/`
`run_manifest.json`, `dataset_manifest.json`, `model_card.json`, `config.yaml`, `environment.json`, `tuning_results.csv`,
`feature_ablation_cv.csv`, `per_matrix_test.csv`, `summary_overall.csv`, `summary_by_group.csv`, `confusion_matrix.csv`,
`paired_comparisons*.csv`, `amortised_cost.csv`, `break_even.csv`, `tolerance_sensitivity.csv`, `exp1_*.csv`, `exp6_*.csv`,
`benchmark_summary.csv`, `figures/`, `report.md`, `TEST_EVALUATED.json` (one-shot guard), `model.joblib` (git-ignored).
