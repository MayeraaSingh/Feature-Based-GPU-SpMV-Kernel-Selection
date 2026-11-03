# SpMV Benchmarking Protocol

This protocol defines the benchmarking methodology and execution standards for collecting GPU SpMV execution timings on the CUDA-capable platform.

## Benchmark Execution Standards

To produce reliable, reproducible, and noise-resilient runtime measurements:

### 1. Warm-Up & Repetitions
- **Warm-Up:** Execute a minimum of 10 unmeasured warm-up iterations to load kernels, initialize driver state, and avoid cold-cache transients.
- **Measurement:** Measure at least 50 to 100 repetitions per matrix-kernel pair using CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`).
- **Metric Aggregation:** Record median execution time in milliseconds (`kernel_time_ms`) to resist sporadic OS background interruptions.

### 2. GPU Environment Control
- Run benchmarks when the GPU is in an idle state with no other compute or graphics workloads active.
- Fix GPU clock frequencies where supported (`nvidia-smi -lgc`) or log GPU temperature, power state, and clock speeds to detect thermal throttling.
- Record hardware and driver specs (GPU model, compute capability, CUDA driver version, CUDA toolkit version).

### 3. Verification Protocol
- Prior to recording performance timings, verify that the computed output vector $y_{gpu} = A \cdot x$ matches the CPU reference vector $y_{cpu}$ within acceptable numerical limits:
  $$\| y_{gpu} - y_{cpu} \|_{\infty} \le \epsilon \cdot (\| A \|_{\infty} \| x \|_{\infty} + 1)$$
- Set `correctness_pass` to `True` only when all values pass numerical verification; otherwise, record `False` and log failure diagnostics.

### 4. Logging & Contract Compliance
- Benchmark executables must write outputs adhering strictly to the contract defined in `docs/data_dictionary.md` (`benchmarks/kernel_timings.csv`).
- Each run should be verifiable and repeatable using automated driver scripts in `scripts/`.

