# Experiment Design

Research question: *Can structural features of a sparse matrix predict which GPU CSR SpMV kernel is fastest, how well does the
prediction generalise to unseen matrix families, and does the benefit survive the cost of obtaining the features and running the
selector?* This document was written before the final analysis; the order of operations below is enforced in code
(`experiment/run.py`).

## Frozen design
* Data: 122 real matrices, frozen split (`docs/dataset_provenance.md`). Candidates: `csr_scalar`, `csr_vector`, `csr_adaptive`, `cusparse`.
* Benchmark: `docs/benchmark_protocol.md` (20 warm-up, 100 timed repetitions, graph timing, round-robin).
* Order: (1) tie tolerance from repetition noise of train+val; (2) feature ablation by grouped CV on train+val; (3) fit and tune every
  method on train (fit) and validation (tuning); (4) **one** scoring pass on the test split, guarded by a one-shot flag file
  (`TEST_EVALUATED.json`); (5) statistics, overhead, sensitivity on that table.
* Methods compared on the same test matrices: `fixed_<kernel>` x4, `fixed_best_train`, `majority_label_train`, `rowlen_heuristic`
  (up to two thresholds on mean row length fitted on train timings), `decision_tree`, `rf_original` (the existing Member 3 Random
  Forest on the seven contract features, tuned on validation), an optional `rf_<set>` if grouped CV prefers another feature set,
  and the `oracle` (upper bound only).

## Experiments
1. **Kernel characterisation** (descriptive, all matrices): wins, geometric-mean slowdown, correlation of relative slowdown with
   size and irregularity features, winners by group.
2. **Feature-based selection**: loss versus the oracle, accuracy (with the measured tie tolerance), within 5/10/25 %, total time ratio.
3. **Generalisation**: results by group and per matrix; group-level bootstrap CIs; claims limited by the number of groups.
4. **Feature ablation**: size-only, size + mean, row-distribution, original 7, extended 11 (four added features, see
   `data_pipeline/features_extended.py`); grouped CV on train+val; the test split is not used.
5. **Overhead**: feature-gathering time (host pass over `indptr`, and a device pass), selector inference time (single-matrix
   `predict`, wall clock), kernel preprocessing; total cost for N = 1 ... 100,000 products and break-even N versus the best fixed kernel.
6. **Repeatability**: the whole benchmark executed twice in separate processes; relative differences, winner agreement, fraction of
   matrices where the winner is not distinguishable from the runner-up.

## Statistics
Matrices of one family are not independent, so the unit of inference is the **group**: cluster bootstrap CIs, group-level paired
sign-flip tests of per-matrix loss differences, Holm correction across the family of comparisons, effect sizes (`dz`) and the
attainable p-value floor are reported. With 7 test groups the floor is 2/2^7 = 0.016 before correction, so most corrected p-values
cannot be significant even for a consistent difference. This is reported, not hidden.

## Not done / out of scope
* Two-stage selector (gather cheap features first, expensive ones only when useful): the measured gathering cost is so small
  relative to the other costs (see results) that the experiment is documented as a cost analysis rather than a separate selector.
* Measurement-based calibration: optional, not implemented (see `docs/limitations.md`).
