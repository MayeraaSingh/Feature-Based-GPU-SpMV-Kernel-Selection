# Results

Run `results/run_real_001` (primary benchmark) and `results/run_real_001_repeat` (independent repeat). Every number comes from the
CSV files in those directories (`results/run_real_001/report.md` lists them all). **Measured findings**, **hypotheses** and
**limits** are labelled separately. Read `docs/limitations.md` with this page.

## 0. Setup in one paragraph
122 real Matrix Market matrices (441 to 90,449 rows, 20 thousand to 4.4 million nonzeros, 23 groups), frozen split of 86 train /
18 validation / 18 test matrices (8 / 8 / 7 groups). Four candidates: three custom CUDA kernels (`csr_scalar`, `csr_vector`,
`csr_adaptive`) and cuSPARSE, float32 on an RTX 3050 6GB Laptop GPU, CUDA 13.1. Kernel time = median of 100 repetitions of the
per-SpMV time (CUDA-graph timing, warm L2). All 488 kernel runs in each of the two benchmark runs passed the correctness check
(largest error 4.4 % of the tolerance, so epsilon = 1e-5 is adequate on this data); there were no failed runs.

## 1. Measurement reliability first (Experiment 6) - measured
* Between the two independent runs the median kernel time differs by a median of **7.9 %** (90th percentile 36 %); run 2 was
  3.5 % faster on median (systematic drift, cause not identified; clocks were stable and no throttling was reported).
* The noise depends strongly on size: median run-to-run difference 13.6 % for matrices under 50 k nonzeros, 7.3 % for
  50-500 k, **2.8 % for >= 500 k** (90th percentile 8.7 %).
* The **winning kernel is the same in both runs for only 75.4 %** of the 122 matrices. For 50 % of matrices the runner-up is within
  the 8 % tie tolerance of the winner, 42 % have a gap above 10 %, 17 % above 25 %.
* **Consequence:** for roughly half of the matrices "the fastest kernel" is not a well-defined label on this machine. Accuracy
  against the oracle is therefore a noisy metric; oracle-relative loss is more robust and is used as the primary metric.
  Differences of a few percent between kernels must not be interpreted. Typical kernel times are 3-30 microseconds (the largest
  matrices 0.2-0.5 ms), so the results describe the small-to-medium, cache-resident regime.

## 2. Kernel performance (Experiment 1, all 122 matrices, descriptive) - measured
| kernel | wins | median time | geometric-mean slowdown vs best | geometric-mean speedup vs cuSPARSE |
|---|---|---|---|---|
| csr_adaptive | 46 | 6.8 us | 1.085 | 1.50 |
| csr_vector | 31 | 7.7 us | 1.256 | 1.30 |
| csr_scalar | 28 | 9.1 us | 1.600 | 1.02 |
| cusparse | 17 | 10.3 us | 1.627 | 1.00 |

* No single kernel wins everywhere: every kernel wins on at least 17 matrices, and inside two of the three large families (`fidap`,
  `bcsstruc`) all four kernels win somewhere (in `drivcav` three of them do). Selection is therefore a real problem on this dataset.
* The custom fused `csr_adaptive` is the best overall kernel on this GPU for these matrices; **cuSPARSE (default algorithm) is the
  slowest on average** here. This is an engineering finding for small, L2-resident float32 matrices on one laptop GPU; it should not be
  generalised to large matrices (the largest matrix runs ~0.3 ms) or other GPUs. The cuSPARSE `ALG_DEFAULT` was not tuned and its
  preprocessing is not included in this kernel-time comparison.
* Rank correlation of the relative slowdown with irregularity: `csr_scalar` gets worse with larger mean row length (rho 0.68),
  maximum row length (0.82) and row-length variance (0.77); `csr_vector` improves with them (-0.68, -0.54, -0.58): the expected
  qualitative behaviour (scalar for short regular rows, warp-per-row for longer ones).
* Where the adaptive kernel helps: it wins most often on mid-length-row families (`fidap` 21 of 40, `cylshell` 5 of 9). Its first
  multi-launch design won 1 of 122 (see `docs/gpu_kernel_design.md`).

## 3. Selector on the held-out test split (Experiments 2-3) - measured, 18 matrices in 7 groups
| method | accuracy (8 % tie tol.) | mean loss | median | 90th pct | max | total-time ratio |
|---|---|---|---|---|---|---|
| **rf_original** (existing Random Forest, 7 features) | 100 % | **0.9 %** (CI 0.0-1.7 %) | 0.0 % | 2.9 % | 5.9 % | 1.025 |
| decision_tree | 100 % | 1.8 % (0.0-2.5 %) | 1.9 % | 3.9 % | 5.9 % | 1.030 |
| rowlen_heuristic | 88.9 % | 156.6 % (0.6-560 %) | 0.0 % | 7.7 % | **2791 %** | 1.261 |
| rf_extended_11 (chosen by grouped CV) | 94.4 % | 8.3 % (0.6-35.4 %) | 0.4 % | 5.1 % | 124 % | 1.087 |
| fixed_best_train (= fixed csr_adaptive) | 66.7 % | 10.3 % (3.3-30.1 %) | 2.8 % | 36.2 % | 45.0 % | **1.018** |
| fixed_csr_vector | 44.4 % | 49.7 % | 8.3 % | 129 % | 194 % | 1.105 |
| fixed_cusparse | 38.9 % | 31.7 % | 10.5 % | 87 % | 106 % | 1.063 |
| fixed_csr_scalar | 61.1 % | 189.4 % | 2.6 % | 218 % | 2791 % | 3.005 |
| oracle (upper bound, not deployable) | 100 % | 0 | 0 | 0 | 0 | 1 |

(Loss = selected time / oracle time - 1, kernel time only. CIs are 95 % bootstrap intervals over **groups**, not matrices.
`majority_label_train` and `fixed_csr_adaptive` coincide with `fixed_best_train`.)

* **Measured:** the Random Forest chose a kernel within 6 % of the oracle on every test matrix (mean loss 0.9 %), against 10.3 % for
  the best fixed kernel chosen on train. The decision tree is almost as good (1.8 %). The simple row-length heuristic is good on
  most matrices (median loss 0 %) but fails catastrophically on one: `gemat1` (max row length 4,230 against mean 9.5) is sent to
  `csr_scalar`, 28x slower than the best kernel. The Random Forest and the tree, which see `max_row_nnz` and `variance_row_nnz`, avoid
  it. This is the clearest benefit of the row-distribution features.
* **Measured but weaker than it looks:** by total runtime (dominated by the few large `cylshell` matrices) the selector is **not**
  better than the best fixed kernel: ratio to oracle 1.025 (selector) against 1.018 (fixed adaptive). The mean-loss advantage comes
  from many small matrices where absolute time differences are microseconds.
* **Large matrices only** (>= 100 k nonzeros, 10 matrices): RF mean loss 0.7 %, fixed adaptive 2.6 %, total-time ratio 1.026 vs 1.011.
* **Per group** (`summary_by_group.csv`, `figures/fig06_per_group.png`): the RF loss is below 6 % in every one of the 7 test groups;
  four groups contain a single matrix, so group-level statements are anecdotal.
* The extended feature set (11 features), which grouped CV on train+validation preferred strongly (group-mean CV loss 1.5 % vs
  6.5 % for the original seven), **did not generalise**: on test it was worse than the original selector because of one matrix
  (`af23560`, loss 124 %). This is a reminder that cross-validated gains on 16 groups can be fragile; the original Random Forest
  remained the pre-declared primary method and is the one reported as the selector.

## 4. Statistical analysis - measured
Paired comparison of `rf_original` with each baseline on the same 18 matrices (loss difference, negative = selector better), tested
at group level (7 groups; sign-flip permutation test on group means; Holm correction over 8 comparisons):

| versus | mean group diff | 95 % CI (group bootstrap) | dz | p | p (Holm) |
|---|---|---|---|---|---|
| fixed_best_train | -17.2 pp | -30.1 to -2.3 pp | -0.98 | 0.031 | 0.22 |
| fixed_csr_vector | -81.7 pp | -122.9 to -15.8 pp | -1.14 | 0.016 | 0.13 |
| fixed_cusparse | -50.2 pp | -75.0 to -12.8 pp | -1.26 | 0.031 | 0.22 |
| fixed_csr_scalar | -158 pp | -566 to -11 pp | -0.46 | 0.125 | 0.38 |
| rowlen_heuristic | -135 pp | -559 to -0.2 pp | -0.39 | 0.25 | 0.50 |
| decision_tree | -0.3 pp | -1.6 to 0.0 pp | -0.43 | 0.50 | 0.50 |

* The effect sizes against the fixed kernels are large and the bootstrap intervals exclude zero, but with **7 independent groups the
  smallest attainable p-value is 0.0156** and after Holm correction **no comparison is statistically significant** (smallest 0.125).
  The Random Forest is **not distinguishable** from the decision tree. We therefore do not claim a statistically significant
  improvement; we report a consistent, large-effect advantage in mean loss that this dataset cannot confirm formally.
* A plain matrix-level bootstrap would give narrower intervals but treats correlated matrices as independent; it is not used for
  conclusions.

## 5. Feature ablation (Experiment 4, grouped 5-fold CV on train + validation: 104 matrices, 16 groups; test not used) - measured
| feature set | features | CV mean loss | CV group-mean loss | CV accuracy |
|---|---|---|---|---|
| size_only | 3 | 7.4 % | 24.8 % | 80.8 % |
| basic_size | 4 | 7.4 % | 23.8 % | 80.8 % |
| row_distribution | 4 | 44.9 % | 49.3 % | 66.3 % |
| original_7 | 7 | 4.5 % | 6.5 % | 79.8 % |
| extended_11 | 11 | 3.3 % | 1.5 % | 83.7 % |

Row-distribution statistics alone are poor predictors (they carry no size information), but adding them to the size features
(original 7) lowers the cross-validated loss from 7.4 %/23.8 % to 4.5 %/6.5 %, and the four extra features lower it further. The
group-mean numbers are dominated by a few small groups with a large loss; the intervals are wide (e.g. size_only 3.7-20.6 %).
See section 3 for why the extended set was not adopted as the primary selector.

## 6. Overhead and amortisation (Experiment 5, test split) - measured
* **Kernel preprocessing:** none for scalar/vector; `csr_adaptive` bins rows on the host (median 0.092 ms including allocation and
  upload, range 0.04-1.4 ms); cuSPARSE descriptors, workspace and `SpMV_preprocess` a median 0.030 ms (0.017-0.165 ms). These are
  roughly 5-50 times a single SpMV (3-30 us). The 0.5 us recorded for scalar/vector is the wall-clock measurement floor, not real work.
* **Feature gathering:** one pass over the row pointer costs a median 0.013 ms on the host (max 0.41 ms) and about 0.10 ms on the GPU
  including the result copy; size features are free.
* **Selector inference (Python, scikit-learn, one matrix, wall clock):** median **6.2 ms for the Random Forest**, 0.34 ms for the
  decision tree, 6.5 us for the heuristic. This is hundreds to thousands of times the kernel time.
* **Consequence (total cost of N products, relative to an N-aware oracle without overhead):**

| N | rf_original | decision_tree | rowlen_heuristic | fixed_best_train |
|---|---|---|---|---|
| 1 | 130x | 9.8x | 5.8x | 6.3x |
| 10 | 14.4x | 1.9x | 1.7x | 1.5x |
| 100 | 2.36x | 1.10x | 1.28x | 1.05x |
| 1,000 | 1.16x | 1.035x | 1.26x | 1.02x |
| 100,000 | 1.027x | 1.030x | 1.261x | 1.018x |

  At no N within 10^7 does any selector beat the best fixed kernel including its own overhead (break-even: **never**), because
  the fixed adaptive kernel is already within 1.8 % of the oracle in total time. For a **single** multiplication the selector is
  the worst option by two orders of magnitude. **Selection pays off only if the selection cost were far lower than measured
  here AND the benefit were concentrated where kernel time matters**; with this dataset the total-time benefit is negligible.
* **Hypothesis (not measured):** a compiled or in-kernel implementation of the decision tree could reduce selection cost to a few
  microseconds; the cost of gathering features (0.013 ms) would then be the remaining overhead. The two-stage idea (gather cheap
  features first, expensive ones only when needed) is not worth building at this overhead ratio: the dominant cost is selector
  inference, not feature gathering.

## 7. Sensitivity - measured
* **Tie tolerance:** RF accuracy is 72 % at 0 %, 78 % at 1 %, 83 % at 2 %, 94 % at 5 % and 100 % at 10 % (decision tree 33 -> 100 %).
  Accuracy is therefore largely a matter of the tolerance; because half of the matrices have runner-up gaps within the measured
  noise, accuracy should not be used to rank selectors. Mean loss does not depend on the tolerance.
* **Large matrices only:** same ranking (section 3).

## 8. What the results show, and what they do not
**Show (measured):** (1) no kernel dominates; (2) a feature-based selector (RF or tree on structure-only features) picks near-oracle
kernels on this test split and avoids the catastrophic choices of a threshold rule; (3) the end-to-end cost of obtaining the
selection (especially Python Random Forest inference) exceeds the kernel time by orders of magnitude, so the selector is not
a practical win for single or few multiplications, and in total time it does not beat a well-chosen fixed kernel.
**Do not show:** statistically significant superiority (7 test groups); generalisation across *distributions* in a broad sense
(23 mostly finite-element / fluid / structural groups, no power-law graphs, test dominated by one family, `cylshell`); results for
large matrices, other GPUs, other precisions, or non-warm caches; that the adaptive kernel is optimal (thresholds untuned).
**Hypotheses:** cheaper selectors would change the overhead conclusion; skewed graph matrices would increase the value of selection.

## 9. Process disclosures (see also `docs/limitations.md`)
* The adaptive kernel was redesigned (multi-launch -> fused) after a first full benchmark whose aggregate results included
  test-split timings; the adoption decision itself used train+validation matrices only.
* The tie-tolerance definition was corrected (from single-repetition IQR to run-to-run difference of medians) after a
  development run of the analysis pipeline on the *repeat* benchmark had shown test-split numbers. The new definition uses only
  train+validation matrices, only affects the accuracy metric, and made the tolerance stricter (8 % vs 16 %). No method, feature
  set, hyper-parameter grid or split was changed after seeing test results. The official run is `run_real_001`; its test split was
  evaluated exactly once.

## 10. Figures (`results/run_real_001/figures/`)
01 kernel comparison, 02 kernel time vs nnz, 03 relative slowdown vs row-length variance, 04 selector vs baselines, 05 loss
distribution, 06 per-group, 07 confusion matrix, 08 feature ablation, 09 break-even / amortised cost, 10 repeatability.
