# Implementation Audit

Audit of the repository as found at commit `5a5f0f6` (branch `main`), performed before any GPU work.
All work since then is on branch `research-development`; `main` is untouched.

## 1. Baseline

| Item | Value |
|---|---|
| Git state | clean working tree, 6 commits (scaffold, Member 2 pipeline, Member 3 selector/evaluation, two merges) |
| Python | 3.13.7; numpy 2.3.3, scipy 1.16.2, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.6 |
| Baseline test run | **267 passed, 0 failed, 1 warning** (164 s) |
| Warning | pandas `FutureWarning` (chained assignment) raised by the test helper in `tests/test_evaluation_e2e.py` (the test, not the library) |
| GPU | RTX 3050 6GB Laptop, compute capability 8.6, driver 592.82 |
| CUDA | at audit time: toolkit absent (`nvcc` not found). Installed afterwards (13.1.115, MSVC 14.44, CMake 4.3.1 + Ninja); see `docs/environment_setup.md` |
| Existing GPU code | none (`gpu/` held only READMEs); `benchmarks/kernel_timings.csv` was header-only |
| Existing dataset | 7 synthetic pilot matrices (`data_pipeline/pilot_data.py`), no real matrices |

## 2. What was already implemented (preserved)
* **Member 2** (`data_pipeline/`): loader (.mtx/.npz), CSR validator, 7 structural features, catalog, group-stratified split generator, CPU float64 reference SpMV + tolerance check.
* **Member 3** (`selector/`, `evaluation/`): dataset-contract validation, oracle and tie handling, Random Forest selector with validation-only tuning, fixed/majority baselines, metrics (loss, accuracy, within-x%), percentile bootstrap, tables, plots, leakage guards; CLI scripts.
No existing implementation was replaced.

## 3. Findings

| # | Finding | Severity | Status |
|---|---|---|---|
| F1 | **Group split ignored group size.** A single large group could be assigned to val or test and leave train tiny. Reproduced on a 59-matrix case with one 30-matrix group: with seed-dependent outcomes between train/val/test = 9/12/38 and 41/9/9 for a 70/15/15 target; most of 40 seeds were far off. | High | **Fixed** (size-aware greedy packing, `data_pipeline/splits.py`); regression tests `tests/test_group_statistics.py` |
| F2 | **Bootstrap assumes independent matrices** (`evaluation.metrics.bootstrap_ci`). Matrices of a family are correlated, so intervals are too narrow. | High (statistical) | **Fixed by addition**: `evaluation/group_stats.py` (cluster bootstrap, group-level paired sign-flip test, Holm correction). Existing functions unchanged and still available; reports use the group-aware ones as primary. |
| F3 | Test helper triggered a pandas `FutureWarning` (chained assignment). | Low | **Fixed** (`tests/test_evaluation_e2e.py`); passes with `-W error::FutureWarning` |
| F4 | `docs/benchmark_protocol.md` instructed to lock clocks with `nvidia-smi -lgc`. The project forbids changing clocks. | Medium (doc) | **Fixed**: protocol rewritten; clocks are only logged |
| F5 | **"Calibration" had two meanings** (README/architecture: confidence calibration; selector_evaluation: measurement-based). | Medium (doc) | **Fixed**: measurement-based only; docs updated |
| F6 | Reference vector x was all-positive (U[0,1)), no cancellation. A kernel with a sign-handling bug would pass. | Medium | **Fixed by addition**: `reference_vector(..., mixed_sign=True)`; the GPU harness uses mixed-sign x. Default unchanged for backward compatibility |
| F7 | Tolerance epsilon = 1e-5 (float32) may be too strict for long rows in principle (worst-case float32 error grows with row length). | Open | **Checked empirically** on the real dataset in `docs/results.md`; epsilon is a harness parameter and all kernels (including cuSPARSE) are checked identically. Nothing is excluded by tolerance without evidence |
| F8 | Tie tolerance (1 %) is an arbitrary default; timing noise on a laptop GPU may exceed it. | Medium | Measured: repeatability experiment reports the noise level and a sensitivity analysis over tolerances is run |
| F9 | `configs/dataset_config.yaml` calls SuiteSparse the intended primary source although it was unreachable (HTTP 403). | Low | Config documents NIST Matrix Market as the acquisition source |
| F10 | README/docs claimed matrix validation rejects explicit zeros / duplicates; behaviour on real files needs checking. | Low | Checked during real dataset ingestion (see `docs/dataset_provenance.md`) |
| F11 | Selector features are known only after a pass over the row pointer; the original contract ignores the **cost of gathering features**. | Research gap | Addressed: measured separately (`overhead_measurements.csv`) and included in end-to-end evaluation |
| F12 | No benchmark harness, kernels, GPU tests, run management or real data. | Missing | Implemented (see other docs) |

## 4. Scientifically questionable points retained (and documented)
* Oracle labels use the single fastest kernel; ties within the tie tolerance are counted as correct in scoring but the training label stays deterministic. Kept (it is the existing documented design) and analysed in a sensitivity study.
* Group definition for splitting is heuristic (`docs/dataset_provenance.md`); related groups may remain across splits. Documented as a threat to validity.
* Random Forest on ~25-35 effective independent groups is statistically weak. This limits any claim and is stated in the results.
