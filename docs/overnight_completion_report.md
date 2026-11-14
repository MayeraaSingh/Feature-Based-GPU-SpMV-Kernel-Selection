# Overnight Completion Report

Branch `research-development` (pushed to `origin`); `main` untouched. No co-author trailers were added anywhere; commits use the
repository's configured identity.

## 1. Status of each objective

| Component | Status | Evidence |
|---|---|---|
| Repository audit | Completed and verified | `docs/implementation_audit.md`; baseline 267 tests passed; 12 findings, fixed or resolved |
| Real dataset | Completed and verified (with scientific limits) | 157 NIST Matrix Market matrices downloaded, SHA-256 verified, 0 failures; 122 after exact-duplicate removal; manifest with URL/date/checksums; `docs/dataset_provenance.md` |
| CSR validation | Completed and verified | `scripts/validate_dataset.py`: 157/157 ok; explicit zeros removed and recorded |
| GPU scalar kernel | Completed and verified | 16 GPU correctness tests + 122 matrices, 0 failures |
| GPU vector kernel | Completed and verified | same |
| GPU adaptive kernel | Completed and verified (thresholds untuned) | same; design history in `docs/gpu_kernel_design.md` |
| cuSPARSE baseline | Completed and verified | ALG_DEFAULT only; matches the float64 reference on all matrices |
| Correctness tests | Completed; sanitizer not run | `tests/test_gpu_correctness.py`; `compute-sanitizer` was not used |
| Benchmarking | Completed and verified | two independent runs, 488 timings each, 0 failures, 0 correctness failures, raw data kept |
| Selector integration | Completed and verified | existing Random Forest, `build_dataset` contracts, extra baselines, one-shot test guard |
| Generalisation evaluation | Completed, **statistically weak** | 18 test matrices / 7 groups; no corrected p < 0.05 |
| Statistical analysis | Completed | cluster bootstrap, group sign-flip tests, Holm, effect sizes, p-floor reported |
| Research report | Completed | `docs/results.md`, `docs/limitations.md`, `results/run_real_001/report.md`, 10 figures |
| Reproduction workflow | Completed | `docs/reproducibility.md`; scripts `check_environment`, `acquire_dataset`, `validate_dataset`, `run_pipeline`, `build_gpu`, `run_gpu_benchmark`, `run_experiment` |
| Two-stage selector | Not completed (by design) | not motivated by measured costs, see `docs/results.md` section 6 |
| Measurement-based calibration | Not completed | optional; documented as not implemented |
| Nsight profiling | Blocked by an external permission | `ERR_NVGPUCTRPERM`, needs administrator action (below) |

## 2. Work by phase
A audit and fixes (split sizing, group statistics, mixed-sign reference vector, warning, protocol, terminology) - B dataset
(crawler, selection rules, validation, duplicate curation, frozen split) - C kernels (4 candidates + multilaunch variant) -
D correctness tests - E harness (CUDA graphs, inner-iteration calibration, round-robin, overheads, telemetry, environment) -
F selector (existing RF plus decision tree, row-length heuristic, ablation) - G evaluation and statistics - H run management
(run ids, manifests, one-shot guard, refusals) - I tests - J documentation - K results - L cleanup.

## 3. Key findings (details and caveats: `docs/results.md`)
1. No kernel dominates; the custom fused adaptive kernel is best on average, cuSPARSE (default algorithm) is slowest on average for these small matrices.
2. Feature-based selection gives near-oracle kernels (mean loss 0.9 % vs 10.3 % for the best fixed kernel) on the test split, but not significantly after correction (7 groups), and not better in total runtime.
3. The selector's inference cost (6.2 ms Python RF) exceeds kernel times by 2-3 orders of magnitude; no break-even versus the best fixed kernel within 10^7 products.
4. Measurement noise is large: same winner in two runs for only 75 % of matrices; accuracy depends mainly on the tie tolerance.
5. Row-distribution features prevent a catastrophic mistake of the row-length heuristic (28x slowdown on one matrix); a cross-validated extended feature set did not generalise to the test split.

## 4. Process disclosures
* Adaptive kernel redesigned after seeing aggregate results (train+val used for the adoption decision).
* Tie-tolerance definition corrected after a development run exposed test numbers; method, features, grids and split unchanged.
* One analysis test failure was committed (`58507cf`) and fixed in the next commit (`829e8e0`).

## 5. Tests
Full suite: **332 passed, 0 failed** (267 original + 65 new), FutureWarnings treated as errors, GPU tests run on the RTX 3050.

## 6. Git
* Branch: `research-development`; remote `origin` (https://github.com/MayeraaSingh/Feature-Based-GPU-SpMV-Kernel-Selection.git).
* Commits (all pushed): `83d2f6a` audit and fixes; `9d14f07` dataset acquisition/validation/frozen split; `14ea3ad` CUDA kernels and harness;
  `58507cf` experiment driver, baselines, overhead, figures; `829e8e0` tolerance rounding fix; `b0d462f` experiment guard tests;
  `345e8c3` results and limitations; plus the final commit containing this report.
* Excluded from Git: raw matrices (`data/raw`, licence unclear, re-fetchable), build outputs (`gpu/build`), cached GPU inputs,
  model binaries, scratch runs (`results/_*`, `results/design_*`).
* Local-only scratch: `results/_design1_multilaunch_all` (first benchmark with the multi-launch adaptive kernel) and
  `results/design_adaptive_trainval` (the design comparison); their conclusions are written in `docs/gpu_kernel_design.md`.

## 7. Remaining blockers / not done
* Nsight Compute: enable GPU performance-counter access (administrator): NVIDIA Control Panel -> Desktop -> Enable Developer Settings
  -> Manage GPU Performance Counters -> allow access to all users.
* A larger, more diverse dataset (power-law graphs, larger matrices) needs an accessible source (SuiteSparse was unreachable).
* `compute-sanitizer` run, cuSPARSE algorithm sweep, adaptive-threshold study on train data, measurement-based calibration, optimised
  (compiled) selector inference, a second GPU.

## 8. Commands to run next
```bash
python scripts/check_environment.py
python -m pytest tests -q                                   # 332 tests
python scripts/run_experiment.py --run-id run_real_002 --benchmark   # fresh benchmark + new single test pass (new run id)
compute-sanitizer gpu/build/spmv_bench.exe --matrix <file.bin> --id x --out <dir> --reps 5 --warmup 2   # after enabling tools
```
