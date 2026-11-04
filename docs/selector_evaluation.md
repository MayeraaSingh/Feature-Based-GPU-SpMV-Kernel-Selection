# Selector & Evaluation Protocol (Member 3)

This document is the written protocol for the kernel selector and its evaluation:
what is trained, what is measured, how metrics are defined, and which decisions
are still open. If code and this document disagree, that is a bug; please report it.

> **Status:** the pipeline is implemented and tested on **synthetic** data only.
> `benchmarks/kernel_timings.csv` is header-only until Member 1 delivers real GPU
> timings, so there are **no real results yet**. Nothing in this repository
> should be quoted as a finding until the real-data run exists.

---

## 1. Question and design

*Can structural matrix features predict which GPU CSR SpMV kernel is fastest on
matrices the selector has never seen?*

A multi-class classifier (Random Forest) maps 7 structural features to a kernel
name. It is evaluated against:

| Method | Meaning |
|---|---|
| `selector` | The trained model's choice |
| `fixed_<kernel>` | Always use that one kernel (one method per candidate) |
| best fixed | The `fixed_<kernel>` that was cheapest in total on the **train** split |
| `oracle` | The measured fastest correct kernel per matrix, an **upper bound**, not an achievable method |

## 2. Inputs (shared contracts)

Defined in [data_dictionary.md](data_dictionary.md). The selector consumes:

| File | Producer | Used for |
|---|---|---|
| `data/metadata/features.csv` | Member 2 | Model inputs (7 numeric columns) |
| `data/metadata/splits.csv` | Member 2 | `train` / `val` / `test` assignment |
| `data/metadata/matrix_catalog.csv` | Member 2 | `matrix_group` for per-group analysis; cross-check of `rows/cols/nnz` |
| `benchmarks/kernel_timings.csv` | Member 1 | Labels, oracle, all runtimes |

Kernel names are **discovered from the timings file**, never hard-coded.

### Validation (`selector/data.py`)
Every failure raises `DataContractError` naming the offending IDs. Nothing is
silently dropped, repaired or relabelled.

- Exact header match; no blank cells; numeric and finite; counts are whole numbers.
- No duplicate `matrix_id` (features, splits, catalog) and no duplicate
  `(matrix_id, kernel_name)` in timings.
- `split` must be exactly `train`, `val` or `test` (**`validation` is rejected**).
- `kernel_time_ms > 0`, `preprocessing_time_ms >= 0`, `correctness_pass` is strictly `True`/`False`.
- The `matrix_id` sets of features, splits and catalog must be identical; every
  timing row must belong to a known matrix; features and catalog must agree on `rows`, `cols`, `nnz`.
- Joins use `matrix_id` (never row position); output is sorted, so input row order is irrelevant.
- **Candidate coverage:** a matrix lacking a timing row for any candidate kernel is
  **excluded and reported** (never imputed): `excluded_matrices.csv`.
- A header-only timings file is refused with a clear message.

## 3. Oracle and labels (`selector/labels.py`)

- A kernel is **eligible** for a matrix only if `correctness_pass` is True.
- **Oracle kernel** = eligible kernel with the lowest `kernel_time_ms` (exact ties: alphabetical).
  Matrices with no eligible kernel have no oracle: excluded and reported.
- **Tie tolerance** (default 1 %, relative): all eligible kernels within tolerance of the
  fastest are "tied". Picking any tied kernel counts as a correct selection. The
  training label stays the single deterministic oracle kernel.
- **End-to-end oracle** (kernel + preprocessing time) is computed separately and reported alongside.
- Training labels come only from `Dataset.timings_for("train")`.

## 4. Model and training (`selector/model.py`, `selector/train.py`)

- `RandomForestClassifier`, fixed `random_state`, `n_jobs=1`. No feature scaling (trees do not need it).
- Features: `rows, cols, nnz, mean_row_nnz, variance_row_nnz, max_row_nnz, min_row_nnz`
  (configurable). Derived features (e.g. max/mean ratio) are a possible later ablation, validation split only.
- Inputs are selected **by column name**; a missing column or NaN/inf is an error.
- The model can only predict kernels that were the oracle on at least one train matrix.
  Expect this limitation to show up in the confusion matrix.
- **Tuning:** small grid (`n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`),
  fitted on train and scored on **validation**. Ranking: (1) fewest picks of a kernel that
  failed correctness, (2) lowest mean oracle-relative loss, (3) highest accuracy, (4) grid
  position. Loss, not accuracy, is the objective because accuracy ignores how costly a mistake is.
- `model_card.json` records config, seed, classes, library versions, git commit, SHA-256 of
  every input file, and a `synthetic_data` flag. Library versions are stored because Colab and a
  laptop differ; **bitwise-identical results across machines are not promised**.

## 5. Leakage controls

| Control | How it is enforced / tested |
|---|---|
| Training never reads the test split | `train_selector` only requests `train`/`val`; a test replaces the accessors with ones that raise on `"test"` |
| Test data cannot influence the model | A test corrupts test timings and test features; tuning table, chosen parameters and predictions on train rows are identical |
| Labels use train timings only | `training_labels` asks for `timings_for("train")` only; tamper test |
| Fixed baseline uses train only | `best_fixed_on_train`; tamper test |
| Frozen model at test time | `evaluate_generalization.py` re-hashes the input files and refuses to run if they differ from the model card |
| One test report per run | A second evaluation of the same run folder is refused unless `--overwrite` |
| Tuning on validation only | Grid is scored on `val`; the test split is not loaded in training |

These are strong guards and tests, not a mathematical proof. `Dataset.features` still holds all rows in memory.

## 6. Metrics (`evaluation/metrics.py`)

For matrix *i*: selected kernel *s_i*, kernel time `t_i(k)` (`kernel_time_ms`), oracle time `T*_i`.

| Metric | Definition |
|---|---|
| **Oracle-relative loss** | `loss_i = t_i(s_i) / T*_i - 1` (0 = optimal; 0.5 = 50 % slower). Stored as a **fraction**, plotted as a percent |
| Slowdown | `t_i(s_i) / T*_i = 1 + loss_i` |
| **Selection correct** | `s_i` passed correctness **and** `t_i(s_i) <= T*_i * (1 + tie_tolerance)` |
| **Selection accuracy** | mean of `selection_correct` over scored matrices. Always read it next to loss: with near-tied kernels, accuracy can be mediocre while loss is near zero |
| Mean / median / p90 / max loss | over matrices with a defined loss |
| 95 % CI | percentile bootstrap (2000 resamples) of the mean loss, resampling matrices |
| Within X % | share of matrices with loss <= 0, 5, 10, 25 % |
| Total time ratio | `sum(selected) / sum(oracle)`; dominated by large matrices, unlike the per-matrix mean |
| End-to-end loss | the same formula on `kernel + preprocessing` time, against the end-to-end oracle |
| Selection overhead | wall-clock `predict` time on **this CPU** per matrix (see limitations) |

**Incorrect selection (Q4, default policy).** If the selector picks a kernel that failed
correctness on that matrix, the pick is a **failure**: `selection_correct = False`, the loss is
`NaN`, and it is counted in `n_incorrect_selection` rather than hidden in the mean. Accuracy
counts it as wrong; loss statistics skip it and report how many were skipped.

Averages are per matrix (one vote per matrix, not weighted by runtime). Empty sets give `NaN`, not errors.

## 7. Running it

```bash
# one-time setup (from the project root)
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q

# real data (needs Member 1's benchmarks/kernel_timings.csv):
python scripts/train_selector.py --config configs/selector_config.yaml
python scripts/evaluate_generalization.py --config configs/selector_config.yaml

# synthetic demo (works today; outputs are stamped SYNTHETIC):
python scripts/make_synthetic_demo.py
python scripts/train_selector.py --config data/synthetic_demo/demo_config.yaml --allow-synthetic --run-id demo
python scripts/evaluate_generalization.py --config data/synthetic_demo/demo_config.yaml --allow-synthetic --run-dir results/demo
```

Training prints the run folder and the exact evaluation command to run next.
CLI flags `--features --splits --catalog --timings --results-dir` override the config paths.

Exit codes: `0` ok, `1` data/config error, `2` synthetic input without `--allow-synthetic`,
`3` test report already exists, `4` input files changed since training.

## 8. Outputs (`results/<run_id>/`, `run_id = YYYYMMDD-HHMM-<git hash>`)

| File | Content |
|---|---|
| `model.joblib` | Trained model (**gitignored**; pickle: only load files you made) |
| `model_card.json`, `tuning_results.csv` | Provenance and validation grid results |
| `per_matrix_test.csv` | One row per (matrix, method) with all per-matrix columns |
| `summary_overall.csv/.md` | One row per method; accuracy, loss stats, CI, ratios |
| `summary_by_group.csv/.md` | Selector vs best fixed per `matrix_group` |
| `confusion_matrix.csv/.md` | Oracle kernel (rows) vs selected kernel (columns) |
| `excluded_matrices.csv`, `train_excluded_matrices.csv` | Matrices removed, with reason |
| `run_manifest.json` | Split evaluated, counts, best fixed kernel, hashes, versions |
| `figures/*.png` | per-matrix loss, loss ECDF, group loss, confusion matrix, total runtime, feature importance |
| `RESULTS_ARE_SYNTHETIC.txt` | Present only for synthetic runs |

Synthetic runs are stamped four ways: a `provenance` column in every CSV, a banner in the
Markdown tables, a banner plus watermark on every figure, and the marker file.

## 9. Limitations (read before quoting numbers)

- **Small n.** With tens of matrices and 2-3 classes a forest has high variance, and the
  test split may be a handful of matrices. Always quote n and the CI; per-group CIs need >= 2 matrices.
- **Unseen classes.** A kernel that never wins on train can never be predicted.
- **Selection overhead is incomplete.** Only CPU `predict` time on the analysis machine is
  measured. Feature extraction (Member 2) is not timed and a laptop CPU is not comparable to GPU time.
- **Group-held-out splits** make the test groups structurally new. This is intentional, but it
  also means weak results may reflect missing training coverage, not a failed method.
- **Feature importances** are impurity-based and diagnostic only.
- **Tie tolerance (1 %)** is a default that has not been compared against real timing noise.

## 10. Decisions log and open questions

| # | Decision / question | Status |
|---|---|---|
| Q1 | Split label is `val` (shared contract), not `validation`; others are rejected | **Implemented** |
| Q2 | After tuning, refit on train+val or keep train-only? | **OPEN.** Switch `selector.refit_on_train_val` exists, default `false`. Decide **before** the test run |
| Q3 | Headline loss uses `kernel_time_ms`; end-to-end reported alongside | **Implemented**, awaiting confirmation |
| Q4 | Selecting an incorrect kernel = failure, NaN loss, counted separately | **Implemented** (default), awaiting confirmation |
| Q5 | Exact `kernel_name` strings, candidate set, repetitions/aggregation, cuSPARSE or not | **OPEN, Member 1.** Code discovers names from data |
| Q6 | Branch name | Resolved: work stays on `selector-evaluation` |
| Q7 | Track small result tables/figures in Git | **OPEN**: proposal is yes, real-data runs only; never models |
| T1 | Tie tolerance 1 % | Default; revisit with Member 1's noise estimate |
| T2 | Duplicate `(matrix_id, kernel_name)` rows are an error, not a median | Default; Member 1 to confirm protocol |
| T3 | A missing candidate-kernel row excludes the matrix (reported) | Implemented |
| T4 | `val` vs `validation` in the original brief | Contract wins (`val`) |

### Terminology: two meanings of "calibration"
`README.md` and `docs/architecture.md` describe **confidence calibration** (probability
calibration, ECE, reliability diagrams, low-confidence fallback). The Member 3 brief defines
**measurement-based calibration**: run a subset of candidate kernels on the new matrix and
count that cost as overhead. The implementation (Milestone 6, optional, not started) follows
the brief. **Members 1 and 2 must agree on this wording** before it is built; the shared docs
have deliberately not been edited.

## 11. Optional calibration study (not implemented)

Planned as a **replay simulation**: with no GPU on the analysis machine, "measuring a kernel on a
new matrix" is a lookup of an already-measured time, and the policy may only use the rows it "paid"
for. If the top-1 confidence is below a threshold, measure the top-m kernels and use the fastest.
Threshold and m are chosen on validation. Overhead is `sum(preprocessing + r * kernel_time)` over
calibration runs (r = repetitions per Member 1's protocol), added to end-to-end time. The report
must state that it is a replay, and "calibration does not pay off" is a valid finding.
