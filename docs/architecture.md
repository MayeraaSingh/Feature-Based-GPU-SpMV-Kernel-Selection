# System Architecture & Data Flow

This document outlines the end-to-end data flow and architectural interfaces for the project: **Evaluating the Generalization of Feature-Based GPU SpMV Kernel Selection Across Sparse Matrix Distributions**.

## End-to-End Data Flow

The project is structured around clear interface contracts and decoupled stages to allow parallel development across CUDA and non-CUDA workstations:

```
[Sparse Matrix Dataset (Raw)]
             │
             ▼
[CSR Loading & Validation]  ────────► (CPU Reference Output)
             │                                   │
             ├──────────────────────┐            │ (Validation)
             ▼                      ▼            ▼
   [Feature Extraction]    [GPU Kernel Benchmarking]
             │                      │
             │                      ▼
             │              [kernel_timings.csv]
             │                      │
             ▼                      │
  [Selector Model Training] ◄───────┘
             │
             ▼
[Held-Out Generalization Evaluation]
             │
             ▼
   [Optional Calibration]
```

---

## Detailed Pipeline Stages

### 1. Sparse Matrix Ingestion & CSR Validation (Member 2)
- Ingests sparse matrices from sources such as SuiteSparse or synthetic matrix generators in standard interchange formats (e.g., Matrix Market `.mtx`).
- Validates data integrity (zero-indexing, sorted indices, absence of explicit zeros or duplicate entries).
- Transforms data into canonical Compressed Sparse Row (CSR) format: `row_ptr`, `col_idx`, `values`.
- Populates `data/metadata/matrix_catalog.csv`.

### 2. CPU Reference & Correctness Baseline (Member 2)
- Computes deterministic CPU reference SpMV solutions ($y = A \cdot x$) using `scipy.sparse` and a custom CPU reference implementation.
- Provides test fixtures for validating GPU kernel correctness.

### 3. Feature Extraction & Dataset Splitting (Member 2)
- Computes structural matrix features (dimensions, non-zero statistics, variance, sparsity ratios).
- Writes structural characteristics to `data/metadata/features.csv`.
- Partitions the catalog into training, validation, and held-out test distributions (e.g., in-distribution vs. out-of-distribution matrix groups), recording assignments in `data/metadata/splits.csv`.

### 4. GPU Kernel Benchmarking (Member 1)
- Implements and compiles multiple CUDA CSR SpMV kernels (scalar, vector, adaptive, and cuSPARSE baseline).
- Benchmarks execution time and preprocessing overhead across all curated matrices on the CUDA-capable machine.
- Validates output vectors against the CPU reference within floating-point tolerance thresholds.
- Logs timing records to `benchmarks/kernel_timings.csv`.

### 5. Selector Model Training (Member 3)
- Joins `features.csv`, `splits.csv`, and `kernel_timings.csv` by `matrix_id`.
- Identifies optimal (oracle) kernels for each matrix based on runtime.
- Trains predictive models (e.g., decision trees, ensembles, multi-class classifiers) on the training split using matrix features as input.

### 6. Held-Out Generalization Evaluation (Member 3)
- Evaluates trained selector models against the held-out test split containing unseen matrix structures and distributions.
- Computes evaluation metrics:
  - Top-1 accuracy against the oracle choice.
  - Fraction of optimal performance (speedup over default heuristic / cuSPARSE).
  - Generalization gap across different matrix distribution classes.

### 7. Uncertainty Estimation & Calibration (Member 3)
- Calibrates probability estimates of kernel optimality to provide confidence scores for model selections.
- Explores fallback heuristics when confidence falls below calibrated decision thresholds.

