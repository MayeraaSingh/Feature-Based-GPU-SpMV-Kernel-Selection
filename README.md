# Evaluating the Generalization of Feature-Based GPU SpMV Kernel Selection Across Sparse Matrix Distributions

## Research Overview

Sparse Matrix–Vector Multiplication (SpMV, $y = A \cdot x$) is a fundamental computational kernel across scientific computing, graph processing, machine learning, and numerical simulations. While modern GPUs provide immense compute throughput and memory bandwidth, SpMV performance is notoriously irregular due to diverse sparsity patterns, non-zero distributions, and resulting thread divergence or memory uncoalescing.

Numerous GPU CSR SpMV kernel variants exist (e.g., CSR-scalar, CSR-vector, CSR-adaptive, cuSPARSE), each excelling under distinct structural conditions. This research investigates whether a feature-based selector trained on lightweight, structural matrix features can reliably choose high-performing SpMV kernels when deployed on unseen sparse matrix distributions.

## Research Objectives

1. **Kernel Benchmarking:** Implement and benchmark multiple GPU SpMV kernels across a varied suite of sparse matrices on an NVIDIA CUDA platform.
2. **Feature Extraction:** Develop a pipeline to extract informative, low-overhead structural features from Compressed Sparse Row (CSR) matrices.
3. **Selector Design & Training:** Train predictive models to recommend optimal SpMV kernels based on matrix features.
4. **Generalization Analysis:** Systematically evaluate whether the selector generalizes across held-out matrix distribution domains (e.g., synthetic vs. real-world graphs vs. regular finite-element meshes).
5. **Confidence Calibration:** Calibrate selector uncertainty to enable fallback mechanisms when faced with out-of-distribution structures.

## High-Level Architecture

The project decouples matrix preprocessing, GPU benchmarking, and selector training via standardized CSV interfaces, enabling parallel development across CUDA and non-CUDA systems:

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

See [docs/architecture.md](file:///c:/Projects/Feature-Based GPU SpMV Kernel Selection/docs/architecture.md) for full architecture details.

## Team Responsibilities

The team comprises three researchers collaborating across one CUDA-capable laptop and two non-CUDA laptops:

| Team Member | Core Responsibilities | Primary Directories |
| :--- | :--- | :--- |
| **Member 1** | GPU kernels (CSR-scalar, CSR-vector, adaptive), CUDA benchmarking harness, runtime logging | `gpu/`, `benchmarks/` |
| **Member 2** | Matrix curation, CSR parsing/validation, CPU reference SpMV, structural feature extraction | `data_pipeline/`, `configs/`, `data/` |
| **Member 3** | Predictive selector modeling, held-out evaluation, metrics, and confidence calibration | `selector/`, `evaluation/` |

## Repository Structure

```text
spmv-kernel-selection/
├── README.md
├── .gitignore
├── requirements.txt
├── LICENSE
│
├── configs/
│   └── dataset_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│       ├── matrix_catalog.csv
│       ├── features.csv
│       └── splits.csv
│
├── data_pipeline/
│   └── README.md
│
├── gpu/
│   ├── README.md
│   ├── kernels/
│   └── benchmarks/
│
├── selector/
│   └── README.md
│
├── benchmarks/
│   ├── README.md
│   └── kernel_timings.csv
│
├── evaluation/
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── benchmark_protocol.md
│
├── tests/
│   └── README.md
│
└── scripts/
    └── README.md
```

## Shared Data Contracts

All components interact via structured CSV contracts documented in [docs/data_dictionary.md](file:///c:/Projects/Feature-Based GPU SpMV Kernel Selection/docs/data_dictionary.md):
- `data/metadata/matrix_catalog.csv`: Inventory of sparse matrices (`matrix_id,rows,cols,nnz,matrix_group`).
- `data/metadata/features.csv`: Extracted matrix properties (`matrix_id,rows,cols,nnz,mean_row_nnz,variance_row_nnz,max_row_nnz,min_row_nnz`).
- `data/metadata/splits.csv`: Train, validation, and held-out partition assignments (`matrix_id,split`).
- `benchmarks/kernel_timings.csv`: Benchmarked execution times and verification records (`matrix_id,kernel_name,kernel_time_ms,preprocessing_time_ms,correctness_pass`).

## Planned Implementation Phases

1. **Phase 1: Foundation & Shared Interfaces (Current)**  
   Repository scaffold, schemas, and contract documentation established.
2. **Phase 2: Data Pipeline & Baseline GPU Kernels**  
   Member 2 implements matrix loading, CSR validation, CPU reference, and feature extraction. Member 1 develops initial CUDA SpMV kernels and verification harnesses.
3. **Phase 3: Benchmarking & Dataset Generation**  
   Member 1 runs benchmark sweeps on the CUDA machine, producing `kernel_timings.csv`. Member 2 generates feature sets and held-out splits.
4. **Phase 4: Selector Training & Generalization Analysis**  
   Member 3 trains predictive selectors on feature data, benchmarks vs. oracle/baseline, and evaluates cross-distribution generalization.
5. **Phase 5: Calibration & Research Synthesis**  
   Member 3 implements uncertainty calibration; team integrates results and prepares project findings.

## Setup Instructions

### Prerequisites
- Python 3.10+
- Optional (for GPU module): NVIDIA GPU with Compute Capability $\ge 6.0$, CUDA Toolkit 11.8+ / 12.0+

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd spmv-kernel-selection

# Create and activate virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

## Contributing and Integrating Work

Because only one laptop has CUDA hardware, all members collaborate through decoupled data contracts:
1. **Branching Strategy:** Work in dedicated feature branches (e.g., `feature/data-pipeline`, `feature/cuda-kernels`, `feature/selector-model`).
2. **Contract Stability:** Do not modify the schemas in `docs/data_dictionary.md` without consensus from all three members.
3. **Data Sharing:** Large raw matrix files (`.mtx`) and binaries are ignored by Git. Share lightweight CSV metadata (`features.csv`, `splits.csv`, `kernel_timings.csv`) through Git to allow Members 2 and 3 to conduct modeling and validation on non-CUDA hardware.
4. **Testing:** Run automated unit tests before opening pull requests:
   ```bash
   pytest tests/
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](file:///c:/Projects/Feature-Based GPU SpMV Kernel Selection/LICENSE) file for details.

