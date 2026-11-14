# Limitations and Threats to Validity

## Statistical power
* **Seven test groups, 18 test matrices.** The exact sign-flip test cannot return p below 2/2^7 = 0.0156, and after Holm
  correction no comparison is significant. Intervals are wide. The data support "a consistent large effect in mean loss", not
  "significantly better".
* **The effective sample is the number of groups (23), not matrices (122).** Three families hold 78 of the 122 matrices; one test
  family (`cylshell`) holds half of the test matrices; four test groups have a single matrix. Per-group numbers are anecdotal.
* **Train has 8 groups, three of them giants** (`fidap`, `bcsstruc`, `drivcav`); it covers a narrow part of the structural space.
  A single frozen split cannot show how results vary with the split; a leave-groups-out study on all data was not done for the
  test question because it would train on test matrices.
* Group definition is a heuristic on Matrix Market set names; related families can still be split across train/val/test.

## Dataset
* NIST Matrix Market only (SuiteSparse unreachable). Mostly finite-element, fluid, structural, economic and physics matrices from
  the 1980s-2000s; **no power-law graphs, no very skewed matrices**; maximum 4.4 million nonzeros. Findings about skewed matrices
  or large matrices cannot be drawn.
* 112 candidates are pattern-only (values set to 1). In the downloaded real/pattern set the values are irrelevant to kernel
  behaviour but the correctness test is less demanding for them.
* Explicit zeros are removed by the existing validator, which changes some patterns relative to the files.
* No explicit licence found on the Matrix Market site; files are not redistributed.

## Measurement
* **Repeatability is the key limit:** run-to-run median difference 7.9 % (13.6 % under 50 k nonzeros), same winner in 75 % of
  matrices, runner-up within 8 % for half. The oracle label is itself noisy; accuracy figures depend strongly on the tie tolerance.
* Kernel times are 3-30 microseconds for most matrices. Per-SpMV times are averages over blocks of back-to-back SpMVs in a CUDA
  graph (K between 1 and a few hundred, recorded), so they include graph-node gaps and **warm L2 cache** behaviour; they are
  not cold-cache or single-call latencies. The timer floor (about 15-20 us per block) is divided by K but not subtracted.
* Laptop GPU under WDDM; clocks not locked (never modified by the project). Telemetry showed no throttling and stable 1.72 GHz
  during the runs, but a systematic 3.5 % drift between the two runs was observed and not explained.
* float32 only; one GPU; one input vector distribution (uniform [-1, 1)); one stream.
* Nsight Compute could not be used (GPU performance-counter permission); no hardware-counter analysis, so explanations of
  kernel behaviour are inferences. `compute-sanitizer` memory checking was not run; correctness relies on the tests, the NaN-poisoned
  output check, and agreement with cuSPARSE and the float64 reference on all 122 matrices.
* Selector inference time is measured in Python/scikit-learn on the CPU; it is an upper-bound style cost, not the cost of an
  optimised implementation.

## Kernels
* `csr_adaptive` thresholds (4/16/64/512, block 128) are fixed design constants, not tuned or shown optimal; bins are not overlapped on
  streams. cuSPARSE was run with its default algorithm only (no CSR_ALG1/ALG2 exploration), so "cuSPARSE is slowest here" refers
  to that configuration.
* The adaptive kernel was redesigned after a first benchmark; a launch-overhead flaw and a local-memory bug were found and fixed
  (`docs/gpu_kernel_design.md`). The redesign decision used train+validation data only but was prompted by aggregate results that
  included the test split.

## Process
* A development run of the analysis on the repeat benchmark exposed test-split numbers before the official run; only a definition
  of the tie tolerance (accuracy metric only) was changed in response (`docs/results.md`, section 9). The official test pass was
  executed once.
* The two-stage selector and measurement-based calibration (optional extensions) were **not implemented**: the first is not
  motivated by the measured cost structure (selector inference dominates, not feature gathering), the second needs a policy study
  that was out of scope; neither is claimed.

## Threats to validity
Internal: timing noise and label noise; unexplained run drift. Construct: oracle = fastest median kernel, which is not well defined
for half of the matrices. External: dataset age/domains, a single GPU, warm-cache small-matrix regime. Statistical: few groups,
multiple baselines compared on one split.
