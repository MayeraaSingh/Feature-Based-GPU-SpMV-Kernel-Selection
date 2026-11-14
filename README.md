# Evaluating the Generalization of Feature-Based GPU SpMV Kernel Selection Across Sparse Matrix Distributions

## Research Overview

Sparse Matrix–Vector Multiplication (SpMV, $y = A \cdot x$) is a fundamental computational kernel across scientific computing, graph processing, machine learning, and numerical simulations. While modern GPUs provide immense compute throughput and memory bandwidth, SpMV performance is notoriously irregular due to diverse sparsity patterns, non-zero distributions, and resulting thread divergence or memory uncoalescing.

Numerous GPU CSR SpMV kernel variants exist (e.g., CSR-scalar, CSR-vector, CSR-adaptive, cuSPARSE), each excelling under distinct structural conditions. This research investigates whether a feature-based selector trained on lightweight, structural matrix features can reliably choose high-performing SpMV kernels when deployed on unseen sparse matrix distributions.

## Research Objectives

1. **Kernel Benchmarking:** Implement and benchmark multiple GPU SpMV kernels across a varied suite of sparse matrices on an NVIDIA CUDA platform.
2. **Feature Extraction:** Develop a pipeline to extract informative, low-overhead structural features from Compressed Sparse Row (CSR) matrices.
3. **Selector Design & Training:** Train predictive models to recommend optimal SpMV kernels based on matrix features.
4. **Generalization Analysis:** Systematically evaluate whether the selector generalizes across held-out matrix distribution domains (e.g., synthetic vs. real-world graphs vs. regular finite-element meshes).
5. **Measurement-Based Calibration (optional extension):** When the selector is unsure, time a small number of candidate kernels on the new matrix and pick the fastest, counting the cost of those measurements. (Earlier drafts also called this "confidence calibration"; the project now uses *calibration* only in this measurement-based sense. Probability calibration of the classifier is not part of the plan.)

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

See [docs/architecture.md](docs/architecture.md) for full architecture details.

## Team Responsibilities

The team comprises three researchers collaborating across one CUDA-capable laptop and two non-CUDA laptops:

| Team Member | Core Responsibilities | Primary Directories |
| :--- | :--- | :--- |
| **Member 1** | GPU kernels (CSR-scalar, CSR-vector, adaptive), CUDA benchmarking harness, runtime logging | `gpu/`, `benchmarks/` |
| **Member 2** | Matrix curation, CSR parsing/validation, CPU reference SpMV, structural feature extraction | `data_pipeline/`, `configs/`, `data/` |
| **Member 3** | Predictive selector modeling, held-out evaluation, metrics, and confidence calibration | `selector/`, `evaluation/` |

## Repository Structure

```text
.
├── configs/            dataset_config.yaml (pilot), dataset_real.yaml (real data), selector_config.yaml
├── data/
│   ├── raw/            matrix files (git-ignored); data/raw/matrix_market/ = NIST downloads
│   └── metadata/       pilot contracts, mm_candidates.csv, dataset_manifest.csv (provenance), real/ (real contracts)
├── data_pipeline/      loader, CSR validator, features (+ extended), splits, CPU reference, acquisition, GPU export
├── gpu/                CUDA C++: kernels/ (scalar, vector, adaptive, cuSPARSE, row stats), benchmarks/ (runner), CMakeLists.txt
├── gpu_bench/          Python orchestration of the benchmark: runner, aggregation, environment capture
├── selector/           Random Forest selector, labels, baselines (Member 3)
├── evaluation/         metrics, group-aware statistics, tables, plots (Member 3)
├── experiment/         methods (tree, heuristic), overhead/amortisation, analysis, run driver, figures, report
├── scripts/            CLI entry points (see docs/reproducibility.md)
├── results/            one directory per run id (benchmark artifacts, tables, figures, report)
├── tests/              pytest suite (GPU tests are skipped, with a reason, when no GPU/executable)
└── docs/               architecture, protocol, data dictionary, audit, kernel design, provenance, results, limitations
```

## Shared Data Contracts

All components interact via structured CSV contracts documented in [docs/data_dictionary.md](docs/data_dictionary.md):
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
5. **Phase 5: Measurement-Based Calibration (optional) & Research Synthesis**  
   Member 3 evaluates measurement-based calibration as a separate, clearly labelled experiment; team integrates results and prepares project findings.

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

### Running the selector and evaluation (Member 3)

The selector and evaluation code needs **no GPU**; it runs on a CPU laptop or in Google Colab. It
consumes the four CSV contracts above. Protocol, metric definitions and open decisions:
[docs/selector_evaluation.md](docs/selector_evaluation.md).

```bash
pytest tests/ -q                                  # all tests (about a minute)

# With real data (needs Member 1's benchmarks/kernel_timings.csv):
python scripts/train_selector.py --config configs/selector_config.yaml
python scripts/evaluate_generalization.py --config configs/selector_config.yaml --run-dir results/<run_id>

# Without real timings: SYNTHETIC demo (outputs are stamped "NOT A RESULT")
python scripts/make_synthetic_demo.py
python scripts/train_selector.py --config data/synthetic_demo/demo_config.yaml --allow-synthetic --run-id demo
python scripts/evaluate_generalization.py --config data/synthetic_demo/demo_config.yaml --allow-synthetic --run-dir results/demo
```

Outputs go to `results/<run_id>/` (tables, figures, model card). A Colab runner is in
[notebooks/colab_run.ipynb](notebooks/colab_run.ipynb). Real GPU timings and results are documented in [docs/results.md](docs/results.md).

> **Terminology ("calibration"):** this project uses *calibration* only to mean **measurement-based
> calibration**: run a few candidate kernels on a new matrix and count that cost as overhead. This
> resolves the earlier inconsistency with "confidence calibration"; see
> section 10 of [docs/selector_evaluation.md](docs/selector_evaluation.md).

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

