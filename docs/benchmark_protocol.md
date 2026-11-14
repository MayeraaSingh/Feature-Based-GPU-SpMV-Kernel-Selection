# SpMV Benchmarking Protocol

This protocol defines how GPU SpMV timings are collected. It describes what the harness in
`gpu/benchmarks/benchmark_runner.cu` and `gpu_bench/` actually does.

## 1. What is measured, and how (never mixed)

| Quantity | Clock | Where recorded |
|---|---|---|
| `kernel_time_ms` | CUDA events around a CUDA-graph launch of the kernel (GPU time, host launch latency excluded) | `kernel_timings.csv` (median), `raw_measurements.csv` (every repetition) |
| `preprocessing_time_ms` | Host wall clock (`steady_clock`) around `prepare()` + `bind()` + `cudaDeviceSynchronize()`; includes device allocation and uploads the kernel needs | `kernel_timings.csv` (median of 10), `raw_measurements.csv` |
| Allocation / host-to-device transfer | Host wall clock (allocation); CUDA events (H2D, median of 5) | `overhead_measurements.csv` |
| Feature gathering (row statistics) | Host wall clock; one pass over host `indptr` (`feature_cpu_ms`) and one device pass including result copy and synchronisation (`feature_gpu_ms`) | `overhead_measurements.csv` |
| Matrix loading / CSR validation / Python feature extraction | Python `perf_counter` | `host_stage_timings.csv` |
| Selector inference | Python `perf_counter` around `predict` | experiment report |
| Timer floor | Same event + graph path around an empty kernel | `overhead_measurements.csv` (`timer_floor_ms`) |

CPU wall-clock numbers are never presented as GPU kernel time. The timer floor is reported, **not subtracted**:
for very small matrices kernel times approach the floor and differences between kernels become unreliable
(see `docs/limitations.md`).

## 2. Warm-up and repetitions
* 20 untimed warm-up launches of every kernel (the protocol minimum is 10), then 100 timed repetitions
  (minimum 50). Both are command-line parameters and recorded in `benchmark_manifest.json`.
* Kernels are captured once into a CUDA graph; each repetition is `event(start) -> graph launch -> event(stop)
  -> synchronise`. Graph replay removes host-side launch latency, which would otherwise dominate small matrices
  and unfairly penalise the multi-launch adaptive kernel and the cuSPARSE API call.
  `--timing direct` (plain launches) is available and is recorded per row in `benchmark_summary.csv`.
* Repetitions are interleaved round-robin over the kernels (rotating start index), so slow thermal / clock drift
  affects all kernels equally.
* Warm-cache protocol: the L2 cache is not flushed between repetitions (representative of repeated SpMV inside
  an iterative solver).
* One matrix per process: a crash or CUDA error in one matrix cannot affect another.

## 3. Aggregation
`kernel_time_ms` is the **median** of the timed repetitions. Mean, standard deviation, IQR, min, max, coefficient
of variation and the number of Tukey (1.5 x IQR) outliers are written to `benchmark_summary.csv`. Outliers are
counted, never removed. Aggregation is a pure function of `raw_measurements.csv` (`gpu_bench/aggregate.py`);
`kernel_timings.csv` is never edited by hand.

## 4. Environment
* Run on an otherwise idle GPU. The harness only *records* temperature, SM and memory clocks, utilisation,
  power, performance state and throttle reasons (`nvidia-smi`) before and after every matrix
  (`gpu_telemetry.csv`) and the GPU, driver, CUDA and compiler versions (`environment.json`).
* **GPU clocks, voltage and power limits are never modified** (`nvidia-smi -lgc` and similar are not used).
  Thermal or power throttling is detected from the telemetry, not prevented.
* The GPU is a laptop part driven through WDDM; clock behaviour is not locked, which adds run-to-run variation.
  Experiment 6 (repeatability) quantifies it.

## 5. Verification (correctness)
For every kernel and matrix, `y_gpu` (float32) is compared with the CPU float64 reference `y_cpu` computed from the
original float64 matrix and the same input vector:

    max_i |y_gpu[i] - y_cpu[i]|  <=  epsilon * (||A||_inf * ||x||_inf + 1),   epsilon = 1e-5

* `||A||_inf` is the maximum absolute row sum of the float64 matrix, `||x||_inf = max|x|` (`data_pipeline/reference_spmv.py`).
* The input vector is **mixed-sign** (uniform in [-1, 1)) so that cancellation is exercised.
* The output buffer is filled with NaN before the check, so a row that a kernel fails to write is detected.
* The check runs on the first (cold) run and again after all timed repetitions. `correctness_pass` is True only
  if **both** pass and the kernel completed without a CUDA error. A non-finite output fails.
* Failed kernels are kept in `kernel_timings.csv` with `correctness_pass = False` (they are excluded from the
  oracle and labels downstream by the selector pipeline). Kernels that did not run to completion have **no** row
  (no zeros, no imputation) and appear in `benchmark_failures.csv` with the reason.
* Error diagnostics (`max_abs_err`, `threshold`, `err_over_threshold`) are in `benchmark_summary.csv`.
  Whether epsilon = 1e-5 is adequate for float32 accumulation over long rows is checked empirically in
  `docs/results.md` (a kernel is never excluded merely by an unjustified tolerance: cuSPARSE is the cross-check).

## 6. Output contract
`kernel_timings.csv`: `matrix_id,kernel_name,kernel_time_ms,preprocessing_time_ms,correctness_pass` (unchanged).
Additional artifacts live next to it: `raw_measurements.csv`, `benchmark_summary.csv`, `overhead_measurements.csv`,
`host_stage_timings.csv`, `gpu_telemetry.csv`, `benchmark_failures.csv`, `environment.json`, `benchmark_manifest.json`.

## 7. Preprocessing and amortisation
`csr_scalar` and `csr_vector` need no preprocessing. `csr_adaptive` bins rows (depends only on the sparsity pattern);
`cusparse` allocates a workspace and runs `cusparseSpMV_preprocess`. Preprocessing is reported separately from kernel
time and the end-to-end oracle (`kernel_time_ms + preprocessing_time_ms`) is computed separately. Preprocessing is
amortisable only if the same sparsity pattern is multiplied repeatedly; the break-even analysis
(`docs/results.md`) reports how many SpMV calls are needed, and never assumes a single setup.
