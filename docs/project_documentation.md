# Project Documentation: Feature-Based GPU SpMV Kernel Selection

A complete reference for the repository: what the project studies, how it is built, how every measurement is taken, how to run it,
what it found, and where its limits are. Every statement about results is backed by files in `results/`; statements that are
hypotheses are labelled as such.

**Status of the evidence (read this first).** The pipeline, kernels and benchmark harness are implemented and tested. The
experimental result is real but statistically weak: 122 real matrices in 23 groups, a held-out test set of 18 matrices in 7 groups,
one GPU. Nothing in this document claims significance, broad generalisation, or novelty.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Background](#2-background)
3. [Research question, scope and non-goals](#3-research-question-scope-and-non-goals)
4. [System architecture](#4-system-architecture)
5. [Repository map](#5-repository-map)
6. [Environment and installation](#6-environment-and-installation)
7. [Data](#7-data)
8. [GPU implementation](#8-gpu-implementation)
9. [Benchmark methodology](#9-benchmark-methodology)
10. [Selector and evaluation methodology](#10-selector-and-evaluation-methodology)
11. [Experiment runner](#11-experiment-runner)
12. [Results](#12-results)
13. [Testing](#13-testing)
14. [Reproducibility](#14-reproducibility)
15. [Reference: command-line tools](#15-reference-command-line-tools)
16. [Reference: configuration files](#16-reference-configuration-files)
17. [Reference: file formats and contracts](#17-reference-file-formats-and-contracts)
18. [Reference: Python and CUDA modules](#18-reference-python-and-cuda-modules)
19. [Troubleshooting](#19-troubleshooting)
20. [Limitations and threats to validity](#20-limitations-and-threats-to-validity)
21. [Decision log and project history](#21-decision-log-and-project-history)
22. [Future work](#22-future-work)
23. [Glossary](#23-glossary)

---

## 1. Executive summary

**What the project does.** It measures how fast four GPU implementations of sparse matrix-vector multiplication (`y = A x`,
`A` in CSR format) run on real sparse matrices, trains a machine-learning *selector* that predicts the fastest implementation from
cheap structural features of the matrix, and evaluates that selector on matrices from families it never saw, including the cost of
obtaining the prediction.

**What was built.**

| Layer | Content |
|---|---|
| Data pipeline (Member 2, extended) | matrix loading, CSR validation, 7 structural features (+4 optional), catalog, group-aware train/val/test split, CPU float64 reference, NIST Matrix Market acquisition with provenance |
| GPU (new) | CUDA C++ kernels `csr_scalar`, `csr_vector`, `csr_adaptive`, cuSPARSE baseline; benchmark runner with CUDA-graph event timing; correctness check against the float64 reference |
| Selector and evaluation (Member 3, extended) | Random Forest selector, fixed/majority baselines, metrics, plus decision tree, row-length heuristic, grouped cross-validation, group-aware statistics, overhead/amortisation analysis |
| Experiment management (new) | run directories, manifests, one-shot test guard, refusal conditions, figures, auto-generated reports |
| Tests | 332 pytest tests (GPU tests run on the real GPU, skipped with a reason when none exists) |

**What it found (run `results/run_real_001`, details in section 12).**

1. No kernel is best everywhere: the fastest kernel is `csr_adaptive` on 46, `csr_vector` on 31, `csr_scalar` on 28 and cuSPARSE on 17 of 122 matrices.
2. A Random Forest on structural features chose a kernel within 5.9 % of the oracle on every one of the 18 test matrices (mean loss 0.9 %), against 10.3 % for the best fixed kernel. The difference is large in effect size but **not statistically significant** after multiple-comparison correction with 7 test groups.
3. Obtaining the prediction costs far more than the kernels it chooses between (about 6 ms in Python against 3-30 microseconds): the selector is 130 times slower than the oracle for a single multiplication, and never beats the best fixed kernel in total runtime.
4. Timing noise is large: across two independent benchmark runs the fastest kernel agrees for only 75 % of matrices, so "the best kernel" is not well defined for about half of the matrices.

**What it does not show.** Significant superiority of the selector; behaviour on power-law graphs or large matrices; results on other GPUs; that the adaptive kernel is optimal.

---

## 2. Background

### 2.1 SpMV and CSR
SpMV computes `y = A x` with `A` sparse. In **Compressed Sparse Row** format the matrix is three arrays: `rowptr` (length
`rows + 1`; row `i` occupies positions `rowptr[i] .. rowptr[i+1]-1`), `colidx` (column of each nonzero) and `val` (its value).
The row length is `len(i) = rowptr[i+1] - rowptr[i]`. SpMV is memory-bandwidth bound: every nonzero is read once together with an
index and an irregular gather from `x`.

### 2.2 Why kernel choice matters
Rows can have very different lengths (empty, a handful, thousands). A GPU kernel must map rows to threads:

* *one thread per row* (scalar) has no communication cost but leaves lanes of a warp reading far-apart memory and suffers when one
  row is much longer than its neighbours;
* *one warp per row* (vector) reads coalesced memory and reduces with shuffles, but wastes lanes on short rows;
* *adaptive* schemes classify rows by length and use different mappings per class, at the price of preprocessing;
* a *vendor library* (cuSPARSE) hides the choice behind an API.

Which is fastest depends on the structure of the matrix. That makes selection a learnable problem: predict the best kernel from
features such as size, mean row length and row-length variance.

### 2.3 Why the cost of selection matters
A selector is only useful if (feature gathering + inference + preprocessing) is smaller than what it saves. Because SpMV on a
small matrix takes microseconds, this cost can dominate, so the project accounts for it explicitly and reports amortised cost over
`N` repeated products.

The project brief names the Seer feature-based selection approach and flat-versus-line-enhanced SpMV kernel selection as
inspiration. This repository does not reproduce or compare against those works and makes no claims about them.

---

## 3. Research question, scope and non-goals

**Question.** *Can structural characteristics of a sparse matrix predict which GPU CSR SpMV kernel will execute fastest, how well
does this generalise to matrix families unseen in training, and does the benefit survive the cost of obtaining and using the
prediction?*

**Objectives.** (1) benchmark several CSR kernels on real matrices; (2) extract low-overhead structural features; (3) train
selectors; (4) evaluate them on held-out *groups* of matrices; (5) account for overhead and preprocessing; (6) quantify
measurement repeatability; (7) optional: measurement-based calibration.

**In scope.** float32 CSR SpMV, one NVIDIA GPU, matrices up to a few million nonzeros, a float64 CPU reference for correctness.

**Non-goals / not done.** Other formats (ELL, COO, HYB), other precisions, multi-GPU, matrices that exceed the 6 GB device,
transposed products, CPU kernels, two-stage selection and measurement-based calibration (documented as not implemented).

**Integrity rules the project follows** (enforced in code where possible): no fabricated or synthetic data in the real experiment;
incorrect kernels are never labels or oracle; test data is never used for training or tuning; the oracle is an upper bound, never
a deployable method; failed runs are recorded, never replaced by zeros; no silent CPU fallback.

---

## 4. System architecture

### 4.1 End-to-end data flow

```
 NIST Matrix Market (web)
        |  scripts/acquire_dataset.py   (crawl, select, download, SHA-256)
        v
 data/raw/matrix_market/*.mtx  +  data/metadata/dataset_manifest.csv   (provenance)
        |  scripts/validate_dataset.py  (checksum, load, CSR validation)
        v
 scripts/run_pipeline.py --config configs/dataset_real.yaml
        |  loader -> validator -> drop duplicate patterns -> features -> split
        v
 data/metadata/real/{matrix_catalog,features,features_extended,splits}.csv   (+ redundancy_log.csv)
        |
        +--------------------------------------------+
        |                                            |
        v                                            v
 scripts/run_gpu_benchmark.py                  CPU float64 reference (reference_spmv.py)
   per matrix: export .bin  -------------------->  y_ref inside the .bin
   spmv_bench.exe (CUDA): prep, correctness, timing, overheads
        |
        v
 results/<run>/benchmark/{kernel_timings.csv, raw_measurements.csv, benchmark_summary.csv, ...}
        |
        v
 scripts/run_experiment.py  (experiment/run.py)
   preflight -> tie tolerance -> feature ablation -> fit/tune (train, val)
   -> ONE test pass -> statistics, overhead, sensitivity -> figures -> report.md
        |
        v
 results/<run>/{per_matrix_test.csv, summary_*.csv, paired_comparisons.csv, amortised_cost.csv, figures/, ...}
```

### 4.2 Design principles
* **Decoupled by CSV contracts.** The four shared files (`matrix_catalog.csv`, `features.csv`, `splits.csv`,
  `kernel_timings.csv`) are the only interface between members, so selector work needs no GPU. Their schemas did not change.
* **One process per matrix** in the GPU benchmark, so a crash or CUDA error cannot contaminate other matrices.
* **Raw data first.** Per-repetition measurements are stored; every aggregate (`kernel_timings.csv`, summaries) is computed from
  them by code, never edited by hand.
* **Leakage guards by construction.** The `Dataset` object hands out timings only per split; training code never requests `test`;
  the test split can be evaluated once per run directory.
* **Failures are data.** A kernel that cannot run is a row in `benchmark_failures.csv`; a matrix lacking a candidate kernel is
  excluded and reported (never imputed).

### 4.3 Responsibilities
| Area | Directory | Origin |
|---|---|---|
| Matrix ingestion, validation, features, split, reference | `data_pipeline/` | Member 2, extended (acquisition, extended features, GPU export, split fix) |
| Kernels and benchmark runner | `gpu/` | new |
| Benchmark orchestration | `gpu_bench/` | new |
| Selector, labels, baselines | `selector/` | Member 3 (unchanged) |
| Metrics, tables, plots | `evaluation/` | Member 3, plus `group_stats.py` (new) |
| Extra baselines, overhead, analysis, runner, figures, report | `experiment/` | new |

---

## 5. Repository map

```
.
|-- README.md                       project overview and quick start
|-- LICENSE                         MIT
|-- requirements.txt                Python dependencies
|-- conftest.py                     pytest path setup
|-- configs/
|   |-- dataset_config.yaml         pilot (synthetic) pipeline configuration
|   |-- dataset_real.yaml           real-data configuration: paths, curation, frozen split seed
|   `-- selector_config.yaml        selector/evaluation configuration (Member 3)
|-- data/
|   |-- raw/                        matrix files (git-ignored); raw/matrix_market/ = NIST downloads
|   |-- processed/                  placeholder
|   `-- metadata/
|       |-- matrix_catalog.csv, features.csv, splits.csv, exclusion_log.csv   pilot (synthetic) contracts
|       |-- mm_candidates.csv       all 498 crawled Matrix Market pages
|       |-- mm_selection.csv        selection decision and reason per candidate
|       |-- dataset_manifest.csv    provenance of every downloaded matrix
|       `-- real/                   real contracts: catalog, features, features_extended, splits, redundancy_log
|-- data_pipeline/
|   |-- loader.py                   .mtx/.npz loading, stable matrix ids
|   |-- validator.py                CSR canonicalisation and checks
|   |-- features.py                 the 7 contract features
|   |-- features_extended.py        4 optional row-distribution features
|   |-- catalog.py                  writers/validators for the contract CSVs
|   |-- splits.py                   group-aware train/val/test assignment
|   |-- pipeline.py                 orchestration (run_pipeline, duplicate-pattern curation)
|   |-- reference_spmv.py           CPU float64 reference and tolerance check
|   |-- acquire.py                  NIST crawler, selection rules, download, checksums
|   |-- gpu_export.py               binary input format for the CUDA runner
|   `-- pilot_data.py               7 synthetic pilot matrices (tests/demos only)
|-- gpu/
|   |-- CMakeLists.txt              library spmv_kernels + executable spmv_bench
|   |-- kernels/
|   |   |-- spmv_common.cuh         interface (SpmvKernel), HostCsr/DeviceCsr, CUDA_CHECK
|   |   |-- subwarp_kernels.cuh     T-threads-per-row and block-per-row device code
|   |   |-- csr_scalar.cu  csr_vector.cu  csr_adaptive.cu  cusparse_baseline.cu  row_stats.cu
|   `-- benchmarks/benchmark_runner.cu   one-matrix benchmark process
|-- gpu_bench/
|   |-- runner.py                   run the executable per matrix, collect outputs
|   |-- aggregate.py                raw samples -> kernel_timings.csv and summaries
|   `-- environment.py              nvidia-smi telemetry, environment capture
|-- selector/                       KernelSelector (Random Forest), labels, baselines, data contracts, train
|-- evaluation/                     metrics, tables, plots, evaluate, group_stats
|-- experiment/
|   |-- methods.py                  TreeSelector, RowLengthHeuristic, tuning, grouped CV
|   |-- overhead.py                 amortised cost and break-even
|   |-- analysis.py                 frozen evaluation plan
|   |-- run.py                      preflight, run_experiment, repeatability
|   |-- figures.py                  10 figures
|   `-- report.py                   report.md generator
|-- scripts/                        command-line entry points (section 15)
|-- tests/                          332 tests (section 13)
|-- results/
|   |-- run_real_001/               primary run: benchmark/, tables, figures, report.md, manifests
|   `-- run_real_001_repeat/        independent repeat of the benchmark
|-- notebooks/colab_run.ipynb       Colab runner for the selector (Member 3)
`-- docs/                           documentation (this file and the specialised documents)
```

Specialised documents: `architecture.md`, `benchmark_protocol.md`, `data_dictionary.md`, `selector_evaluation.md`,
`implementation_audit.md`, `gpu_kernel_design.md`, `dataset_provenance.md`, `experiment_design.md`, `environment_setup.md`,
`reproducibility.md`, `results.md`, `limitations.md`, `overnight_completion_report.md`.

---

## 6. Environment and installation

### 6.1 Reference machine
Windows 11, NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute capability 8.6), driver 592.82, Intel i5-13420H, 16 GB RAM.

| Component | Version | Note |
|---|---|---|
| CUDA Toolkit | 13.1 (nvcc 13.1.115) | `winget install Nvidia.CUDA --version 13.1`; matches the driver's CUDA 13.1 |
| cuSPARSE | 12.7.3 | shipped with the toolkit |
| Host compiler | MSVC toolset 14.44 (cl 19.44.35228) | selected with `VsDevCmd.bat -vcvars_ver=14.44`; 14.51 untested |
| Build | CMake 4.3.1 + Ninja | bundled with Visual Studio; the Visual Studio generator is not used |
| Python | 3.13.7 | numpy 2.3.3, scipy 1.16.2, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.6, PyYAML 6.0.3 |

### 6.2 Setup commands
```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py      # read-only report; exit 0 = ready for GPU work
python scripts/build_gpu.py              # builds gpu/build/spmv_bench.exe
```
CPU-only work (selector training, evaluation, most tests) needs only the Python packages. The GPU path needs an NVIDIA GPU, the
CUDA Toolkit and a C++ host compiler; on Linux, `cmake` and `g++`.

### 6.3 Windows-specific notes
* CUDA 13 keeps its runtime DLLs in `bin\x64`. The installer adds it to the machine `PATH`; terminals opened earlier do not see it and
  programs exit with `0xC0000135`. Open a new terminal. `gpu_bench/runner.py` also prepends the directory for the benchmark subprocess.
* Clocks and power limits are never changed. The laptop GPU runs under WDDM with unlocked clocks; telemetry is recorded.
* Nsight Compute is installed but needs administrator-granted access to GPU performance counters (`ERR_NVGPUCTRPERM`); it was not used.

---

## 7. Data

### 7.1 Source and licence
All matrices are real, from the **NIST Matrix Market** (<https://math.nist.gov/MatrixMarket/>). SuiteSparse returned HTTP 403 for file
downloads when the project started and when re-checked, so it is not used. No explicit licence text was found on the Matrix Market
site (the disclaimer URL returns 404); the repository therefore **does not redistribute** matrices. It stores URLs, dates and
SHA-256 checksums and fetches the files on demand. Cite the original contributors (`source_note`) and Matrix Market when publishing.

### 7.2 Acquisition (`data_pipeline/acquire.py`, `scripts/acquire_dataset.py`)
1. **crawl**: parse the "matrices by name" page (498 matrix pages), then each page for size, type (real/complex/integer/pattern,
   symmetric/...), file URL, set, source and discipline. Two alternative statistics layouts (SPARSKIT and MMSTATS blocks) are parsed
   by `_parse_alt_stats`; for symmetric matrices stored as one triangle the full nonzero count is `2 * strict_lower + diagonal`.
   Result: `data/metadata/mm_candidates.csv`.
2. **select** (rules fixed before any timing existed, see 7.3): `mm_selection.csv` records a decision and reason for each candidate.
3. **download**: each file is fetched with retries and a polite delay, gunzipped, written to `data/raw/matrix_market/<id>.mtx`, and
   hashed (`sha256_gz`, `sha256_mtx`). The manifest row records source URL, licence note, acquisition date and status.
4. **verify**: re-hashes all local files against the manifest.

### 7.3 Selection rules
1. Numeric field `real`, `integer` or `pattern` (complex excluded: the benchmark is real float32). Pattern matrices have no stored
   values; Matrix Market convention (value 1) is used. Their structure is real; their values are not.
2. Expanded nonzeros at least 20,000, because smaller matrices run in about a microsecond and are dominated by timer/launch floors.
   Outcome: 157 matrices selected, 157 downloaded, 0 failures.
3. The CSR validator removes explicit zeros, so the canonical `nnz` can be smaller than the file's count (`nnz_csr` in the manifest).
4. **Exact duplicate sparsity patterns are dropped** (SHA-256 of shape, `indptr` and `indices`; the smallest `matrix_id` is kept;
   each drop is logged in `data/metadata/real/redundancy_log.csv`). 35 were dropped (31 in `drivcav`), leaving **122 matrices**.
5. Nothing was excluded because it was inconvenient for the GPU or because of its timings.

### 7.4 Validation (`data_pipeline/validator.py`, `scripts/validate_dataset.py`)
`validate_csr` converts to CSR, sorts column indices within rows (required by cuSPARSE), sums duplicate `(i, j)` entries, removes
explicit zeros, and checks dimensions, pointer length, index ranges and nnz consistency. Empty rows are valid; a fully empty matrix is
an error. `validate_dataset.py` additionally checks that the file exists and matches its SHA-256, loads it through the project loader,
writes `validation_status`, `nnz_csr` and `sparsity` back to the manifest, and refuses synthetic manifest rows. Result: 157/157 ok.

### 7.5 Dataset characteristics (122 matrices)
441 to 90,449 rows (median about 3,600); 20 thousand to 4.4 million nonzeros; mean row length 3.9 to 173; maximum row length up to
4,230; only 6 matrices contain empty rows. Domains are mostly finite-element/structural, fluid dynamics, economics and plasma
physics. There are **no power-law web/social graphs**, which limits conclusions about highly skewed matrices.

### 7.6 Groups
Group = Matrix Market set name with trailing digits and an `_old` suffix removed (`bcsstruc1..5` -> `bcsstruc`, `drivcav_old` ->
`drivcav`): a heuristic for "related matrices". 23 groups; three hold most matrices: `fidap` 40, `bcsstruc` 23, `drivcav` 15.

### 7.7 The split (`data_pipeline/splits.py`)
Whole groups are assigned to train, validation or test (70/15/15 target) so related matrices never straddle a boundary.

*Algorithm (size-aware greedy packing).* Groups are shuffled with the seeded generator (random tie-breaking), then stably sorted from
largest to smallest. Each group goes to the split minimising the squared error between the resulting split sizes and the target
sizes (an optional group-count term is available, `GROUP_COUNT_WEIGHT`, default 0). When there are at least three groups, a split
that would stay empty receives a group.

*Why it changed.* The original implementation assigned shuffled groups to test, then validation, ignoring size; on a 59-matrix case with
one 30-matrix group it produced train/val/test between 9/12/38 and 41/9/9. Adding a group-count term was tried (weights 0.25, 0.5, 1.0) to
give train more families, but it shrank validation and test to 5-12 matrices, so it is off. Only sizes and structure informed these
choices, never timings.

*Frozen split* (seed **20261115**): train 86 matrices (fidap, bcsstruc, drivcav, sherman, psmigr, astroph, dwave, econaus), validation 18
(econiea, tokamak, mhd, lshape, crystal, dwt, hamm, saylor), test 18 (cylshell 9, gemat 3, lns 2, airfoil, bcspwr, mvmrwk, platz).
`data/metadata/real/splits.csv` is committed; never regenerate it after looking at test results.

### 7.8 Features
Original contract (`data_pipeline/features.py`; row lengths from `indptr`, population variance, empty rows included):

| Feature | Definition |
|---|---|
| `rows`, `cols`, `nnz` | dimensions and canonical nonzero count |
| `mean_row_nnz` | `nnz / rows` |
| `variance_row_nnz` | population variance (ddof 0) of row lengths |
| `max_row_nnz`, `min_row_nnz` | extremes of row length |

Optional extended features (`data_pipeline/features_extended.py`, separate `features_extended.csv`, contract unchanged):

| Feature | Definition | Question it answers |
|---|---|---|
| `cv_row_nnz` | `sqrt(variance) / mean` | scale-free irregularity |
| `empty_row_frac` | fraction of rows with 0 nonzeros | wasted threads |
| `max_over_mean` | `max / mean` row length | heavy-tail indicator |
| `frac_rows_gt_32` | fraction of rows with more than 32 nonzeros (32 = warp width) | rows that can fill a warp |

All features come from `indptr` only, so they are available before any GPU timing. Timings are never selector inputs.

### 7.9 The pilot data
Seven synthetic matrices (`pilot_data.py`) and `data/metadata/*.csv` exist for tests and demos. They are never mixed with the real data
(real artifacts live in `data/metadata/real/` and `results/`), and tools refuse inputs whose name contains "synthetic" unless explicitly
allowed.

---

## 8. GPU implementation

### 8.1 Common interface (`gpu/kernels/spmv_common.cuh`)
All candidates consume the same device CSR (`int32 rowptr`, `int32 colidx`, `float32 val`), the same `x` and write the same `y` (float32 compute).

```cpp
class SpmvKernel {
  virtual std::string name() const = 0;
  virtual void prepare(const HostCsr&, const DeviceCsr&, cudaStream_t) = 0;  // timed preprocessing
  virtual void bind(const float* x, float* y, cudaStream_t) {}               // once, untimed pointers (cuSPARSE descriptors)
  virtual void run(const float* x, float* y, cudaStream_t) = 0;              // the timed SpMV: no sync, no allocation
  virtual void release() = 0;
};
```
`run()` performs no synchronisation or allocation so that it can be captured into a CUDA graph. `CUDA_CHECK` turns any CUDA error
into a C++ exception that the runner records as a failure.

### 8.2 `csr_scalar` (one thread per row)
```
for r in grid-stride rows:                  # 256 threads per block
    s = 0
    for k in rowptr[r] .. rowptr[r+1]-1:  s += val[k] * x[colidx[k]]     # x through __ldg
    y[r] = s
```
No reduction, no synchronisation, no preprocessing. Neighbouring threads read different rows, so loads of `val`/`colidx` are strided; long rows serialise
on one thread and cause divergence.

### 8.3 `csr_vector` (one warp per row)
Device routine `tpr_body<32>`: a group of T=32 threads (one warp) handles one row; lane `l` processes nonzeros `beg + l, beg + l + 32, ...`
(consecutive lanes read consecutive memory: coalesced), then a shuffle-down tree (`__shfl_down_sync` with offsets 16, 8, 4, 2, 1) leaves the sum
in lane 0, which writes `y[row]`. All 32 lanes always execute the shuffles (no early return); out-of-range groups contribute 0. 128 threads per
block. Weakness: idle lanes for short rows.

### 8.4 `csr_adaptive` (row bins, single fused launch)
**Preprocessing (host, separately timed):** scan `rowptr`, classify each row by length `L`, build one row-index list per bin, allocate and upload.

| Bin | Rows | Mapping | Rationale |
|---|---|---|---|
| 0 | `L <= 4` (empty rows included: they write 0) | 1 thread per row | at most 4 nonzeros per thread |
| 1 | `L <= 16` | 4 threads per row | at most 4 per thread |
| 2 | `L <= 64` | 16 threads per row | at most 4 per thread |
| 3 | `L <= 512` | 32 threads (warp) per row | at most 16 per lane |
| 4 | `L > 512` | 128-thread block per row (shuffle + 4-entry shared memory) | block-level parallelism |

**Execution:** one kernel (`k_adaptive_fused`) whose grid is the concatenation of the bins' blocks. Each block finds its bin from the cumulative
block offsets (a block-uniform branch, so no divergence) and runs the corresponding device routine. No atomics: every row is written by exactly one
thread group. The parameter struct is passed `const __grid_constant__` (see 8.8). The thresholds are **fixed design constants, not tuned and not
claimed optimal**; bins are not overlapped on streams. The preprocessing depends only on the sparsity pattern, so it amortises only if the pattern is
reused.

### 8.5 cuSPARSE baseline (vendor library)
`cusparseSpMV` with CSR, int32 indices, float32 values and compute type, `CUSPARSE_SPMV_ALG_DEFAULT`, non-transpose. `prepare()` sets the stream
(the handle is created once per process, as an application would, and is not counted); `bind()` builds the descriptors, queries the buffer size,
allocates the workspace and calls `cusparseSpMV_preprocess`; all of that is counted as preprocessing. The timed `run()` is the `cusparseSpMV` call only.
Only the default algorithm is exercised. It is a vendor library, never presented as a custom kernel.

### 8.6 Row statistics kernel (`row_stats.cu`)
A device pass over `rowptr` computing sum of squares, maximum and minimum row length (warp shuffles plus three atomics per warp). Used only to measure
the cost of gathering row-distribution features on the GPU and to check that it agrees with the CPU pass.

### 8.7 Build (`gpu/CMakeLists.txt`, `scripts/build_gpu.py`)
Static library `spmv_kernels` (kernels, links `CUDA::cudart` and `CUDA::cusparse`) and executable `spmv_bench`. C++17; `CMAKE_CUDA_ARCHITECTURES=native`
unless overridden; Release by default. On Windows `build_gpu.py` locates CUDA (`CUDA_PATH`) and Visual Studio (`vswhere`), writes a temporary batch file
that runs `VsDevCmd.bat -vcvars_ver=<toolset>`, prepends the CUDA, CMake and Ninja directories, and calls CMake with Ninja and
`-DCMAKE_CUDA_HOST_COMPILER=cl`.

### 8.8 Design history (reported honestly)
1. **Multi-launch adaptive** (one launch per non-empty bin) won on 1 of 122 matrices in a first full benchmark: at microsecond runtimes the extra
   launches dominated. This prompted a redesign (the aggregate included test-split timings).
2. **Fused adaptive.** A first fused version passed its parameter struct by value and ran 3-5 times *slower*: `ptxas -v` showed an 88-byte per-thread stack
   frame (the struct copied to local memory). `const __grid_constant__` removed the slowdown. On the 104 train+validation matrices the
   fused kernel was faster than the multi-launch version on 104 of 104 (geometric-mean time ratio 0.63) and was adopted on that basis. The multi-launch
   variant remains in the executable as `csr_adaptive_multilaunch`, correctness-tested, but is not a selector candidate.

### 8.9 Input format (`data_pipeline/gpu_export.py`)
Little-endian binary: magic `SPMVBIN1`; `int64` rows, cols, nnz; `float64` `||A||_inf` and `||x||_inf`; `int32 indptr[rows+1]`; `int32 indices[nnz]`;
`float32 values[nnz]` (float32 rounding of the float64 values); `float32 x[cols]`; `float64 y_ref[rows]`. `x` is mixed-sign, uniform in [-1, 1),
seed 42, float32-representable. `y_ref` is the float64 CPU SpMV of the ORIGINAL float64 matrix, so the correctness check also covers the float32
rounding of the values. Matrices beyond int32 range are refused.

### 8.10 Runner flow (`gpu/benchmarks/benchmark_runner.cu`), one matrix per process
1. Read the problem; check that about 1.5 times the needed device memory is free, otherwise exit with code 4 (recorded as `insufficient_gpu_memory`).
2. Allocate device buffers (wall-clock); upload (CUDA events, median of 5).
3. Measure the timer floor (empty-kernel graph, 120 launches, first 20 discarded).
4. Measure feature gathering: host pass over `indptr` and device pass including result copy and synchronisation (55 iterations, first 5 discarded);
   verify that the device result equals the host result.
5. For each kernel: one untimed warm preparation; `prepare()+bind()+sync` timed 10 times (host wall clock); poison `y` with NaN, run once, compare
   (first correctness check); calibrate the inner-iteration count `K`; capture the graph of `K` back-to-back SpMVs.
6. 20 warm-up launches of every kernel, then 100 timed repetitions interleaved round-robin with a rotating start.
7. Poison `y` again, run, compare (second correctness check). Write per-kernel status, raw samples and auxiliary measurements.
Exit codes: 0 ok, 1 fatal error, 2 usage/unknown kernel, 3 unreadable input, 4 not enough GPU memory.

---

## 9. Benchmark methodology

### 9.1 What is measured and with which clock
| Quantity | Clock | Stored in |
|---|---|---|
| `kernel_time_ms` | CUDA events around a CUDA-graph launch (GPU time, host launch latency excluded) | `kernel_timings.csv` (median), `raw_measurements.csv` |
| `preprocessing_time_ms` | host wall clock around `prepare()+bind()+cudaDeviceSynchronize()` | same |
| allocation, H2D transfer | wall clock / CUDA events | `overhead_measurements.csv` |
| feature gathering (host and device) | wall clock | same |
| matrix load + validation, Python feature extraction, export | Python `perf_counter` | `host_stage_timings.csv` |
| selector inference | Python `perf_counter` around `predict` | experiment tables |
| timer floor | events + graph around an empty kernel | `overhead_measurements.csv` |

These are never mixed; CPU wall-clock numbers are never presented as kernel time.

### 9.2 CUDA graphs and inner iterations
A single launch measured with events contains a fixed overhead (the timer floor was measured at 15-50 microseconds) that is comparable to the kernels
themselves. Each kernel's `run()` is therefore captured into a graph containing `K` back-to-back SpMVs and each repetition measures
`event -> graph launch -> event`, reporting `block time / K`. `K` is calibrated per kernel from a pilot (16 single launches, median of the last 11) so
that a block lasts about 1 ms, clamped to [1, 1000], and recorded as `inner_iters`. The result is a per-SpMV time with the fixed overhead divided by `K`.
The measurement therefore includes graph-node gaps (which penalise multi-kernel designs) and **warm L2 cache** behaviour (the matrix stays resident between
consecutive SpMVs). A plain-launch mode (`--timing direct`) exists.

### 9.3 Repetitions and aggregation
20 warm-up launches (protocol minimum 10) and 100 timed repetitions (minimum 50); repetitions are interleaved round-robin over the kernels. Median,
mean, standard deviation, IQR, min, max, coefficient of variation and the number of Tukey outliers (1.5 x IQR) are computed from the raw data in
`gpu_bench/aggregate.py`; outliers are counted, never removed. `kernel_timings.csv` holds the median kernel time and the median preprocessing time.

### 9.4 Correctness
For every kernel and matrix: `max_i |y_gpu[i] - y_ref[i]| <= eps * (||A||_inf * ||x||_inf + 1)` with `eps = 1e-5`, `||A||_inf` the maximum absolute row sum,
checked on the first (cold, NaN-poisoned) run and again after the timed repetitions; non-finite output fails. `correctness_pass` is true only if both checks
pass and the kernel ran without a CUDA error. In the real runs the largest error was 4.4 % of the threshold, so `eps = 1e-5` is adequate for this data (rows
up to 4,230 nonzeros). A fault-injection option (`--inject-fault <kernel>`) proves that a wrong result is detected.

### 9.5 Failure handling
A kernel that fails to run gets **no** row in `kernel_timings.csv` (no zeros, no imputation) and a row in `benchmark_failures.csv` (matrix, kernel, stage,
reason). A kernel that runs but is wrong gets a row with `correctness_pass = False` (excluded from oracle and labels downstream). Downstream, a matrix lacking
any candidate kernel is excluded and reported.

### 9.6 Environment logging
`environment.json` (GPU, driver, CUDA path, nvcc, package versions, git commit/branch/dirty, start telemetry) and `gpu_telemetry.csv` (temperature, SM and
memory clocks, utilisation, memory, power, P-state, throttle reasons before and after every matrix). In the two real runs: 56-71 degrees C, SM clock stable at
about 1.72 GHz, throttle reasons 0x0 throughout.

### 9.7 Known measurement biases
Warm-cache small matrices; graph-node gaps; WDDM; unlocked laptop clocks; float32 only; a single input vector distribution; single stream. See section 20.

---

## 10. Selector and evaluation methodology

### 10.1 Labels, oracle, ties (`selector/labels.py`)
A kernel is *eligible* for a matrix only if `correctness_pass` is true. The **oracle kernel** is the eligible kernel with the lowest `kernel_time_ms`
(exact ties broken alphabetically). Every eligible kernel within the *tie tolerance* of the fastest is counted as a correct selection when scoring; the
training label stays the single deterministic oracle kernel. A separate end-to-end oracle minimises `kernel_time_ms + preprocessing_time_ms`.

### 10.2 Measurement-based tie tolerance (`experiment/analysis.py`)
Kernels closer than the measurement noise cannot be ranked. The tolerance is the median, over train+validation matrices, of
`|median_run2 - median_run1| / median_run1` for the fastest kernel (run-to-run noise of medians from the two independent benchmark runs), rounded up to a
whole percent, minimum 1 %. On this data: **8 %** (median relative IQR of single repetitions 20 %, run-to-run median 7.06 %, 90th percentile 37 %). Without a
repeat run the relative IQR of the winner is used as a documented fallback. Only train+validation matrices are used. Mean loss does not depend on the tolerance.

### 10.3 Methods compared on the same test matrices
| Method | Description |
|---|---|
| `fixed_<kernel>` (4) | always use that kernel |
| `fixed_best_train` | the single kernel with the lowest total time on TRAIN among kernels correct on every train matrix |
| `majority_label_train` | the kernel that is the oracle on most train matrices |
| `rowlen_heuristic` | up to two thresholds on `mean_row_nnz`, one kernel per band, fitted on TRAIN timings by minimising total train time (`RowLengthHeuristic`) |
| `decision_tree` | scikit-learn decision tree on the 7 features; grid `max_depth {2,3,4,6,None}` x `min_samples_leaf {1,2,4}` tuned on validation loss |
| `rf_original` | the existing Member 3 Random Forest (`selector.model.KernelSelector`, `selector.train.train_selector`) on the 7 features; grid `n_estimators {100,300}` x `max_depth {None,4,8}` x `min_samples_leaf {1,2}` x `class_weight {None,balanced}` (24 settings) tuned on validation |
| `rf_<set>` | optional Random Forest on the feature set preferred by grouped CV (here `extended_11`) |
| `oracle` | the measured fastest correct kernel: an upper bound, never a deployable method |

Tuning ranks candidates by (1) fewest selections of a kernel that failed correctness, (2) lowest mean validation oracle-relative loss, (3) highest accuracy,
(4) grid order. `refit_on_train_val` is false (the final model is trained on train only).

### 10.4 Leakage prevention
* `Dataset.timings_for(split)` is the only way to reach timings; training code requests only `train` and `val`.
* Feature-set selection and ablation use grouped cross-validation on train+validation only.
* The test split is scored once; `results/<run>/TEST_EVALUATED.json` makes a second evaluation of the same run directory impossible.
* Splits are by group; the group definition is a heuristic (a threat to validity, see section 20).

### 10.5 Metrics (`evaluation/metrics.py`)
For matrix `i`, selected kernel `s_i`, oracle time `T*_i`:
* loss `= t_i(s_i) / T*_i - 1` (0 = optimal; 0.5 = 50 % slower); slowdown `= 1 + loss`;
* selection correct: the selected kernel passed correctness and its time is within the tie tolerance of the oracle; accuracy = mean of that flag;
* a selection of an incorrect kernel counts as wrong and has an undefined (NaN) loss, reported in `n_incorrect_selection`;
* `total_time_ratio = sum(selected time) / sum(oracle time)` (dominated by large matrices);
* `within_5/10/25 %`: fraction of matrices with loss at most the threshold; mean, median, 90th percentile and maximum loss.

### 10.6 Group-aware statistics (`evaluation/group_stats.py`)
Matrices of one family are not independent, so the unit of inference is the group.
* `cluster_bootstrap_ci`: percentile CI of the mean, resampling whole groups with replacement (fewer than 2 groups gives NaN).
* `group_mean`: mean of per-group means (each group weighs the same).
* `paired_group_signflip_test`: per-matrix differences of loss (method A minus method B) are averaged within groups; a sign-flip permutation test on the
  group means (exact enumeration up to 14 groups, Monte Carlo above) gives a two-sided p-value; also the mean group difference, the standardised effect
  `dz = mean / sd` of group values, a cluster-bootstrap CI and `p_floor`, the smallest attainable p-value (2 / 2^G). With 7 groups it is 0.0156.
* `holm_adjust`: Holm-Bonferroni correction across the family of comparisons.
The original matrix-level bootstrap (`bootstrap_ci`) remains available but is not used for conclusions.

### 10.7 Feature ablation (Experiment 4)
Grouped 5-fold cross-validation over train+validation (104 matrices, 16 groups; folds balance matrix counts, whole groups per fold) for five sets:
`size_only` (rows, cols, nnz), `basic_size` (+ mean_row_nnz), `row_distribution` (mean, variance, max, min), `original_7`, `extended_11`. Reports CV mean loss,
group-mean loss with a cluster-bootstrap CI, and accuracy. The set with the lowest group-mean CV loss (fewer features on ties) defines the optional
`rf_<set>` method. The primary selector remains `rf_original`, declared in advance.

### 10.8 Overhead and amortisation (`experiment/overhead.py`)
For matrix `i` and kernel `k`, `T_ik(N) = preprocessing_ik + N * kernel_time_ik`. For a method `m`, `Total_m(N) = sum_i [selection_overhead_mi + T_{i,k_m(i)}(N)]`;
the N-aware oracle is `sum_i min_k T_ik(N)` over correct kernels, with no overhead. `selection_overhead` = feature-gathering time (host pass; zero for size-only
methods) + selector inference time (single-matrix `predict`, wall clock, Python). Totals are linear in N, so the **break-even** `N` between two methods is solved
exactly (`inf` if the method never catches up). Preprocessing is charged once per matrix: it assumes the sparsity pattern is reused for all N products.

### 10.9 Sensitivity
Accuracy of the frozen selections under tie tolerances 0, 1, 2, 5, 10 % (scoring only, nothing refitted); the large-matrix subset (nnz at least 100,000);
group-mean versus matrix-mean loss.

---

## 11. Experiment runner

`python scripts/run_experiment.py --run-id <id> [--benchmark]` (implementation: `experiment/run.py`).

**Stages.** (optional with `--benchmark`: environment check, dataset validation, pipeline, GPU build, benchmark) -> preflight -> tie tolerance -> feature
ablation -> fit and tune (train, validation) -> set the one-shot flag -> score the test split -> statistics, amortisation, break-even, sensitivity, Experiment 1
characterisation -> provenance files -> figures and `report.md`.

**Refusals** (`RefusedToRun`, exit code 3, nothing evaluated on the test split):
* a required input file is missing (features, splits, catalog, extended features, timings, summary, overhead, failures, manifest, environment record);
* an input file name contains "synthetic", or a manifest row has a synthetic source;
* a CSV contract is violated (the existing validation in `selector/data.py`, e.g. duplicate `(matrix_id, kernel)` rows, bad split labels, id mismatches);
* more than 20 % of matrices lack a candidate kernel (incomplete benchmark);
* a candidate kernel never passed correctness; fewer than two candidate kernels;
* the test split of this run directory was already evaluated.
`--benchmark` additionally stops if the environment check, dataset validation, pipeline or build fails.

**Run directory layout** (`results/<run_id>/`): `config.yaml`, `environment.json`, `dataset_manifest.json`, `run_manifest.json` (git state, seeds, tolerance,
input SHA-256, split counts), `model_card.json`, `tuning_results.csv`, `feature_ablation_cv.csv`, `per_matrix_test.csv`, `summary_overall.csv`,
`summary_by_group.csv`, `summary_large_matrices.csv`, `confusion_matrix.csv`, `paired_comparisons.csv`, `paired_comparisons_extra.csv`, `amortised_cost.csv`,
`break_even.csv`, `tolerance_sensitivity.csv`, `exp1_*.csv`, `exp6_repeatability_*.csv`, `benchmark_summary.csv`, `figures/`, `report.md`,
`TEST_EVALUATED.json`, `benchmark/` (raw artifacts) and the git-ignored `model.joblib`. Previous runs are never overwritten.

---

## 12. Results

Full analysis with caveats: `docs/results.md`; every number is in `results/run_real_001/`. Summary:

### 12.1 Benchmark quality
488 timings per run (122 matrices x 4 kernels), 0 failures, 0 correctness failures in both runs. Median kernel times per kernel 7-10 microseconds (largest matrices 0.2-0.5 ms).
Between the two runs the median kernel time differs by a median of 7.9 % (13.6 % under 50 k nonzeros, 2.8 % above 500 k; 90th percentile 36 %); the winner agrees
on 75.4 % of matrices; for 50 % of matrices the runner-up is within the 8 % tolerance. Run 2 was 3.5 % faster (unexplained drift).

### 12.2 Kernel comparison (Experiment 1, all matrices)
| kernel | wins | median time | geomean slowdown vs best | geomean speedup vs cuSPARSE |
|---|---|---|---|---|
| csr_adaptive | 46 | 6.8 us | 1.085 | 1.50 |
| csr_vector | 31 | 7.7 us | 1.256 | 1.30 |
| csr_scalar | 28 | 9.1 us | 1.600 | 1.02 |
| cusparse | 17 | 10.3 us | 1.627 | 1.00 |

### 12.3 Selection on the test split (18 matrices, 7 groups)
| method | accuracy (8 %) | mean loss (95 % group CI) | median | max | total-time ratio |
|---|---|---|---|---|---|
| rf_original | 100 % | 0.9 % (0.0-1.7) | 0.0 % | 5.9 % | 1.025 |
| decision_tree | 100 % | 1.8 % (0.0-2.5) | 1.9 % | 5.9 % | 1.030 |
| rowlen_heuristic | 88.9 % | 156.6 % (0.6-560) | 0.0 % | 2791 % | 1.261 |
| rf_extended_11 | 94.4 % | 8.3 % (0.6-35.4) | 0.4 % | 124 % | 1.087 |
| fixed_best_train (csr_adaptive) | 66.7 % | 10.3 % (3.3-30.1) | 2.8 % | 45.0 % | 1.018 |
| fixed_cusparse | 38.9 % | 31.7 % | 10.5 % | 106 % | 1.063 |
| oracle (upper bound) | 100 % | 0 | 0 | 0 | 1 |

Paired group-level tests of `rf_original`: mean loss difference -17.2 percentage points against `fixed_best_train` (CI -30.1 to -2.3, dz -0.98, p 0.031, Holm
0.22); no comparison is significant after Holm correction (smallest 0.125); indistinguishable from the decision tree. The heuristic's maximum loss comes from one
matrix (`gemat1`, maximum row length 4,230 against mean 9.5) that it sends to `csr_scalar`, 28 times slower than the best kernel; the models that see
`max_row_nnz` and `variance_row_nnz` avoid it. The extended feature set that grouped CV preferred did not generalise to the test split (one matrix, `af23560`, loss 124 %).

### 12.4 Overhead (test split)
Selector inference (Python, single matrix): Random Forest 6.2 ms, decision tree 0.34 ms, heuristic 6.5 microseconds. Feature gathering: 0.013 ms (host), about 0.10 ms
(device). Preprocessing: adaptive median 0.092 ms, cuSPARSE 0.030 ms. Total cost relative to the N-aware oracle:

| N | rf_original | decision_tree | rowlen_heuristic | fixed_best_train |
|---|---|---|---|---|
| 1 | 130x | 9.8x | 5.8x | 6.3x |
| 100 | 2.36x | 1.10x | 1.28x | 1.05x |
| 100,000 | 1.027x | 1.030x | 1.261x | 1.018x |

No selector beats the best fixed kernel in total cost for any N up to 10^7 (break-even "never").

### 12.5 Ablation (grouped CV on train+validation; group-mean loss)
size_only 24.8 %, basic_size 23.8 %, row_distribution 49.3 %, original_7 6.5 %, extended_11 1.5 %. Row-distribution statistics alone are poor predictors; added to size
features they reduce cross-validated loss substantially.

### 12.6 Interpretation
**Measured:** no kernel dominates; a feature-based selector chooses near-oracle kernels and avoids the catastrophic choices of a threshold rule; selection cost dominates
for small matrices. **Not shown:** statistical significance, broad generalisation, results for large matrices or other GPUs. **Hypotheses:** an optimised (compiled) selector
could cut selection cost to microseconds; skewed graph matrices would increase the value of selection.

---

## 13. Testing

`python -m pytest tests -q` runs 332 tests (97 s on the reference machine, GPU tests included). GPU tests are skipped with an explicit reason when the executable or an
NVIDIA GPU is absent; they are never replaced by CPU emulation.

| File | Tests | Covers |
|---|---|---|
| `test_data_pipeline.py` | 49 | loader, validator, features, catalog, splits, pipeline, reference |
| `test_reference.py` | 20 | CPU reference SpMV and tolerance check |
| `test_contracts.py` | 21 | CSV contract headers and formats |
| `test_selector_data.py` | 46 | data contract validation, joins, leakage accessors |
| `test_labels.py` | 20 | oracle, ties, eligibility |
| `test_selector_model.py` | 37 | KernelSelector, tuning, baselines |
| `test_metrics.py` | 36 | loss, accuracy, bootstrap, scoring |
| `test_evaluation_e2e.py` | 36 | CLI end-to-end on synthetic demo data, corrupt-input errors |
| `test_import_isolation.py` | 2 | package-name shadowing regression |
| `test_group_statistics.py` | 26 | cluster bootstrap, sign-flip test, Holm; split sizing regression |
| `test_methods_overhead.py` | 14 | tree, heuristic, tuning, grouped CV, amortisation, tie tolerance |
| `test_gpu_correctness.py` | 16 | every kernel on edge cases on the real GPU; fault injection; graph vs direct timing; overhead records; error messages |
| `test_experiment_guards.py` | 9 | refusal conditions of the experiment driver |

GPU edge cases: random square, empty rows, single-element rows, wide and tall rectangular matrices, a 200,000-nonzero row inside a 50 x 300,000 matrix, a power-law irregular
matrix and a 1 x 1 matrix, with mixed-sign `x`. **Not covered:** `compute-sanitizer` runs, matrices with zero nonzeros (rejected upstream by the validator), very large matrices.

---

## 14. Reproducibility

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/acquire_dataset.py download --min-nnz 20000      # needs network; crawl output is committed
python scripts/acquire_dataset.py verify
python scripts/validate_dataset.py
python scripts/run_pipeline.py --config configs/dataset_real.yaml
python scripts/build_gpu.py
python -m pytest tests/test_gpu_correctness.py -q
python scripts/run_gpu_benchmark.py --out-dir results/run_real_001/benchmark
python scripts/run_gpu_benchmark.py --out-dir results/run_real_001_repeat/benchmark
python scripts/run_experiment.py --run-id run_real_001
```
Seeds: split 20261115 (`configs/dataset_real.yaml`); model and bootstrap seed 0 (`--seed`). `run_manifest.json` records the git commit, input checksums and tolerance;
`environment.json` the hardware and software versions. A run id evaluates its test split once; repeat the analysis in a new run directory (copy `benchmark/`).
Timings depend on the machine; a different GPU gives different numbers and possibly different winners. The selection pipeline runs on any CPU given a `kernel_timings.csv`
from a real benchmark. Details: `docs/reproducibility.md`.

---

## 15. Reference: command-line tools

| Script | Purpose | Main options |
|---|---|---|
| `check_environment.py` | read-only environment report | `--json FILE` |
| `acquire_dataset.py {crawl,refresh-missing,download,verify}` | Matrix Market crawl, download, verification | download: `--min-nnz`, `--max-nnz`, `--raw-dir` |
| `validate_dataset.py` | checksum + load + CSR validation, updates manifest | `--manifest`, `--raw-dir` |
| `run_pipeline.py` | loader -> validator -> features -> split -> contract CSVs | `--config`, `--raw-dir`, `--pilot`, `--verbose` |
| `build_gpu.py` | configure and build the CUDA executable | `--msvc-toolset` (default 14.44), `--arch` (default native), `--clean` |
| `run_gpu_benchmark.py` | benchmark all (or selected) matrices | `--out-dir` (required), `--config`, `--warmup` (>=10), `--reps` (>=50), `--eps`, `--timing graph|direct`, `--matrix-ids`, `--exe` |
| `run_experiment.py` | full experiment into `results/<run_id>/` | `--run-id` (required), `--config`, `--benchmark`, `--seed`, `--skip-report` |
| `train_selector.py` / `evaluate_generalization.py` | Member 3's original selector workflow on the shared CSV contracts | `--config`, `--features/--splits/--catalog/--timings`, `--run-id`, `--allow-synthetic`, `--overwrite` |
| `make_synthetic_demo.py` | generate SYNTHETIC demo inputs (tests only) | - |

Executable `spmv_bench` (called by the Python harness): `--matrix F --id ID --out DIR [--warmup N --reps N --eps E --prep-reps N --timing graph|direct --kernels a,b --inject-fault K --inner N --target-block-ms X]`.

---

## 16. Reference: configuration files

`configs/dataset_real.yaml`
| Key | Meaning |
|---|---|
| `dataset.raw_dir`, `processed_dir`, `metadata_dir` | locations (`data/raw/matrix_market`, `data/processed`, `data/metadata/real`) |
| `dataset.manifest` | provenance CSV from which `matrix_group` is read |
| `matrix_sources` | informational (NIST enabled, SuiteSparse disabled) |
| `matrix_groups` | optional id -> group overrides |
| `feature_extraction.extended` | also write `features_extended.csv` |
| `curation.drop_identical_patterns` | drop exact duplicate sparsity patterns (logged) |
| `splits.train_ratio/val_ratio/test_ratio/random_seed/stratify_by` | 0.70/0.15/0.15, **20261115**, `matrix_group` |

`configs/selector_config.yaml` (Member 3): `paths.*`, `results_dir`, `selector.feature_columns`, `selector.seed`, `selector.tie_tolerance`, `selector.refit_on_train_val`, `selector.grid`,
`evaluation.bootstrap_samples`, `evaluation.bootstrap_seed`. `configs/dataset_config.yaml` configures the synthetic pilot pipeline.

---

## 17. Reference: file formats and contracts

The four shared contracts (unchanged; full definitions in `docs/data_dictionary.md`):
`matrix_catalog.csv` (`matrix_id,rows,cols,nnz,matrix_group`), `features.csv` (`matrix_id,rows,cols,nnz,mean_row_nnz,variance_row_nnz,max_row_nnz,min_row_nnz`),
`splits.csv` (`matrix_id,split` with `split` in `train|val|test`), `kernel_timings.csv` (`matrix_id,kernel_name,kernel_time_ms,preprocessing_time_ms,correctness_pass`).
Validation rules (`selector/data.py`): no missing values, no duplicate `(matrix_id, kernel_name)`, `kernel_time_ms > 0`, `preprocessing_time_ms >= 0`, consistent ids and `rows/cols/nnz`
across files, no padded kernel names.

Additional artifacts (columns in `docs/data_dictionary.md`): `dataset_manifest.csv`, `mm_candidates.csv`, `mm_selection.csv`, `redundancy_log.csv`, `features_extended.csv`,
and in each `benchmark/` directory `raw_measurements.csv`, `benchmark_summary.csv`, `overhead_measurements.csv`, `host_stage_timings.csv`, `gpu_telemetry.csv`,
`benchmark_failures.csv`, `environment.json`, `benchmark_manifest.json`.

---

## 18. Reference: Python and CUDA modules

**`data_pipeline`**: `load_matrix`, `load_directory` (stable ids from file stems); `validate_csr`; `extract_features`, `extract_extended_features`; `make_splits`; `run_pipeline`,
`drop_identical_patterns`; `cpu_spmv`, `check_result`, `reference_vector(..., mixed_sign)`; `acquire.{build_candidate_table, refresh_missing, select_candidates, download_matrix, verify_manifest_checksums,
group_from_set}`; `gpu_export.{make_problem, write_gpu_input}`.

**`gpu_bench`**: `runner.{exe_path, cuda_env, run_one, benchmark_matrices, KERNELS}`; `aggregate.{describe_samples, aggregate_matrix, aux_row, write_frames}`;
`environment.{gpu_telemetry, capture_environment, git_info}`.

**`selector`**: `data.{build_dataset, Dataset, DataContractError}`; `labels.{oracle_table, training_labels}`; `model.KernelSelector`; `baselines.{best_fixed_on_train, majority_kernel_on_train, fixed_selections}`;
`train.{train_selector, TrainConfig}`.

**`evaluation`**: `metrics.{score_selections, summarize_scores, bootstrap_ci, fraction_within, total_time_ratio}`; `group_stats.{cluster_bootstrap_ci, group_mean, paired_group_signflip_test, holm_adjust}`;
`evaluate.evaluate`; `tables`, `plots` (Member 3 report outputs).

**`experiment`**: `methods.{TreeSelector, RowLengthHeuristic, tune_on_validation, group_folds, grouped_cv_loss, mean_loss}`; `overhead.{method_total, oracle_total, amortised_table, break_even}`;
`analysis.{measure_tie_tolerance, fit_methods, feature_ablation, method_selections, score_all, summarize, paired_comparisons, confusion, tolerance_sensitivity}`;
`run.{preflight, run_experiment, repeatability, RefusedToRun}`; `figures.make_all`; `report.write_report`.

**CUDA**: `make_csr_scalar/vector/adaptive/adaptive_multilaunch/cusparse`, `gpu_row_stats`, device routines `tpr_body<T>` and `block_row_body<BLOCK>`, kernels `k_csr_scalar`, `k_tpr<T>`,
`k_block_row<B>`, `k_adaptive_fused`, `k_row_stats`.

---

## 19. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvcc` not found / build fails at configure | CUDA Toolkit missing or not on PATH: install (section 6), set `CUDA_PATH`, open a new terminal; `python scripts/check_environment.py` |
| benchmark process exits with `3221225781` (`0xC0000135`) | CUDA `bin\x64` not on PATH of the shell; open a new terminal (the Python runner adds it automatically) |
| `ERROR: benchmark executable not found ... build_gpu.py` | run `python scripts/build_gpu.py` |
| `vswhere.exe is not recognized` printed during a build | harmless message from `VsDevCmd.bat`; the toolset is still selected |
| `REFUSED: ... already evaluated` | the run directory's test split was scored; use a new `--run-id` and copy `benchmark/` into it |
| `REFUSED: timing data incomplete` | too many matrices lack a kernel; read `benchmark_failures.csv` |
| `insufficient_gpu_memory` in the failures file | matrix needs more than about two thirds of the free device memory; it is skipped by design |
| `ERR_NVGPUCTRPERM` from Nsight Compute | enable GPU performance counters for all users (administrator, NVIDIA Control Panel developer settings) |
| download returns HTTP 403 (SuiteSparse) | that host blocks downloads; the project uses NIST Matrix Market |
| GPU tests are skipped | no NVIDIA GPU visible or the executable is not built (the skip message says which) |
| timings differ between runs | expected: see section 12.1; compare medians within the noise, never single winners |

---

## 20. Limitations and threats to validity

**Statistical power.** 7 test groups and 18 test matrices: the exact test cannot go below p = 0.0156 and nothing is significant after correction. The effective sample is the number of groups (23);
three families hold 78 of 122 matrices and one test family holds half of the test matrices. Train has 8 groups. A single frozen split cannot show how results vary with the split.

**Data.** One source (NIST, mostly 1980s-2000s finite-element, fluid, structural, economic matrices); no power-law graphs; at most 4.4 million nonzeros; 112 candidate matrices are pattern-only; the validator
removes explicit zeros; group definitions are heuristic; no explicit licence.

**Measurement.** Run-to-run noise of 7.9 % (median) and 75 % winner agreement; microsecond kernels; warm-L2 per-SpMV averages with graph-node gaps; WDDM laptop GPU with unlocked clocks; an unexplained 3.5 % drift between
runs; float32 only; one GPU, one stream, one input-vector distribution; no hardware-counter profiling; no `compute-sanitizer`. Selector inference time is measured in Python.

**Kernels.** Adaptive thresholds untuned; no stream overlap of bins; cuSPARSE default algorithm only; the adaptive kernel was redesigned after seeing aggregate results.

**Process.** A development run exposed test-split numbers before the official run; only the definition of the tie tolerance (accuracy metric) changed afterwards; methods, features, grids and split did not.
One analysis test failure was committed and fixed in the next commit.

**Threats.** Internal: timing and label noise. Construct: "fastest median kernel" is not well defined for half of the matrices. External: dataset domains, single GPU, small cache-resident matrices.
Statistical: several baselines compared on one small split. Full list: `docs/limitations.md`.

---

## 21. Decision log and project history

| Step | Event |
|---|---|
| 1 | repository audit: baseline 267 tests passing; 12 findings (`docs/implementation_audit.md`) |
| 2 | CUDA 13.1, MSVC 14.44, CMake/Ninja set up and verified (vector-add and cuSPARSE test programs executed on the GPU) |
| 3 | group-split sizing bug reproduced and fixed; group statistics added |
| 4 | NIST crawl (498 pages); selection rules; 157 matrices downloaded and verified |
| 5 | duplicate-pattern curation (157 -> 122); split frozen (seed 20261115, 86/18/18) |
| 6 | harness measurement redesign: single launches had a 15-50 us floor, so CUDA graphs with calibrated inner iterations were introduced (noise fell from CV 20-110 % to 2-7 %) |
| 7 | adaptive kernel: multi-launch (1/122 wins) -> fused; parameter-copy bug fixed with `__grid_constant__`; adoption decided on train+validation only |
| 8 | two benchmark runs; analysis pipeline debugged on a development copy; tie-tolerance definition corrected; official single test pass on `run_real_001` |
| 9 | results, limitations, completion report; 332 tests passing |

Commits on `research-development`: `83d2f6a` audit and fixes, `9d14f07` dataset, `14ea3ad` CUDA kernels and harness, `58507cf` experiment driver, `829e8e0` rounding fix, `b0d462f` guard tests,
`345e8c3` results, `e227eff` completion report.

---

## 22. Future work

* A larger and more diverse dataset including power-law graphs and large matrices (needs an accessible source).
* Measurement-based calibration as a separately labelled experiment with full cost accounting.
* An optimised selector (compiled decision tree, or selection fused into the benchmark) to test the overhead hypothesis.
* A study of the adaptive thresholds on training data only; stream overlap of bins; cuSPARSE algorithm sweep.
* More repetitions and a locked-clock or controlled-environment benchmark to tighten run-to-run noise; cold-cache measurements.
* `compute-sanitizer` and Nsight profiling once counter access is enabled.
* Leave-groups-out evaluation over all data (without a held-out test claim) to study split sensitivity.

---

## 23. Glossary

| Term | Meaning |
|---|---|
| SpMV | sparse matrix-vector multiplication `y = A x` |
| CSR | compressed sparse row format (`rowptr`, `colidx`, `val`) |
| nnz | number of stored nonzeros |
| warp | 32 GPU threads executing in lockstep |
| coalesced access | neighbouring threads reading neighbouring memory addresses |
| CUDA graph | a recorded sequence of GPU work that can be replayed with one launch |
| oracle | the measured fastest correct kernel for a matrix; an upper bound, not a deployable method |
| oracle-relative loss | `t_selected / t_oracle - 1` |
| tie tolerance | relative difference below which two kernels count as equally fast (measured, 8 %) |
| group | family of related matrices (set name); the unit of splitting and inference |
| cluster bootstrap | bootstrap that resamples whole groups |
| Holm correction | step-down correction of p-values for multiple comparisons |
| `p_floor` | smallest p-value attainable by the sign-flip test with G groups (2 / 2^G) |
| amortised cost | total cost of `N` products including preprocessing and selection overhead |
| break-even | smallest `N` at which a method's total cost is no larger than another's |
| preprocessing | per-matrix setup a kernel needs before it can run (bins, workspace) |
| timer floor | fixed event/graph overhead measured with an empty kernel |
| leakage | information from the test split influencing training or tuning |
