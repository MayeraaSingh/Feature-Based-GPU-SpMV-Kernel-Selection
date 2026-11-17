# GPU Kernel Design

Four candidates for `y = A x` with `A` in CSR (int32 indices, float32 values), float32 compute, one stream.
All consume the same device CSR arrays, the same `x`, write the same `y`, and are checked against a float64
CPU reference (`docs/benchmark_protocol.md`). Sources: `gpu/kernels/`; common interface `spmv_common.cuh`
(`prepare` = timed preprocessing, `run` = timed multiplication, `release`).

| | csr_scalar | csr_vector | csr_adaptive | cusparse |
|---|---|---|---|---|
| Source | `csr_scalar.cu` | `csr_vector.cu` | `csr_adaptive.cu` | `cusparse_baseline.cu` |
| Type | custom | custom | custom | **vendor library, not a custom kernel** |
| Mapping | 1 thread / row | 1 warp (32 lanes) / row | row bins: 1, 4, 16, 32 threads / row, or a 128-thread block / row | library-chosen (`CUSPARSE_SPMV_ALG_DEFAULT`) |
| Memory pattern | lanes read 32 different rows (uncoalesced `val`/`colidx`); `x` via `__ldg` | consecutive lanes read consecutive `val`/`colidx` (coalesced); `x` via `__ldg` | as vector for the long bins; sub-warp groups for short rows | library internal |
| Reduction | none | 5-step `__shfl_down_sync` tree | sub-warp shuffle (width T); block-per-row: shuffle + 4-entry shared memory | library internal |
| Synchronisation | none | none (no `__syncthreads`) | `__syncthreads` only in the block-per-row branch | library internal |
| Preprocessing | none | none | host scan of `rowptr`, bin lists, allocation + upload (separately timed) | descriptors, workspace query + allocation, `cusparseSpMV_preprocess` (handle created once per process, not counted) |
| Known limitations | divergence and load imbalance on irregular rows; long rows serialise | idle lanes on short rows; one warp per very long row | preprocessing cost; thresholds fixed, not tuned; no stream overlap of bins | black box; algorithm choice not controlled here |

## Adaptive kernel details

Row classes by length `L` (thresholds are fixed design constants, **not tuned on timings and not claimed optimal**):

| bin | rows with | mapping | rationale |
|---|---|---|---|
| 0 | L <= 4 (includes empty rows, which write 0) | 1 thread / row | <= 4 nonzeros per thread |
| 1 | L <= 16 | 4 threads / row | <= 4 nonzeros per thread |
| 2 | L <= 64 | 16 threads / row | <= 4 nonzeros per thread |
| 3 | L <= 512 | warp / row | <= 16 nonzeros per lane |
| 4 | L > 512 | 128-thread block / row | long rows get block-level parallelism |

No atomics are used; every row is written by exactly one thread group.

### Design history (reported honestly)
1. **Iteration 1, `csr_adaptive_multilaunch`:** one kernel launch per non-empty bin. In a first full benchmark of the
   dataset it won on only 1 of 122 matrices: matrices at this size run in a few microseconds, so the extra launches
   (inter-kernel gaps) dominated. This aggregate result (which includes test-split timings) motivated the redesign.
2. **Iteration 2, `csr_adaptive` (the candidate used):** the same strategy fused into **one launch**; the grid is the
   concatenation of the bins' blocks and every block dispatches on its bin (a block-uniform branch).
   * A first version took the parameter struct by value and ran 3-5x *slower* than iteration 1. `ptxas -v` showed an
     88-byte per-thread stack frame (the struct was copied to local memory). Declaring the parameter
     `const __grid_constant__` removed it. This was an implementation bug, not a property of the strategy.
   * The decision to adopt iteration 2 was taken on **train + validation matrices only** (104 matrices): faster than
     iteration 1 on 104 of 104, geometric-mean time ratio 0.63. Iteration 1 stays available in the executable
     (`--kernels csr_adaptive_multilaunch`) and is correctness-tested, but is not a selector candidate.
3. Disclosure: the decision to redesign was made after seeing aggregate benchmark results, including test-split
   timings. Nothing was tuned to individual test matrices; see `docs/limitations.md`.

## Launch configurations
* scalar: 256 threads / block, grid = ceil(rows / 256), grid-stride loop.
* vector and sub-warp bins: 128 threads / block, grid = ceil(rows x T / 128).
* adaptive (fused): 128 threads / block, grid = total blocks of all bins.
* Architecture: CMake builds for `native` (the detected GPU, compute capability 8.6 here); override with
  `python scripts/build_gpu.py --arch 86`.
* No CPU fallback exists anywhere in the harness: if a kernel fails to run it is recorded in
  `benchmark_failures.csv` and the matrix is reported as incomplete.

## Correctness coverage (`tests/test_gpu_correctness.py`, runs on the GPU)
Random square, rows with no nonzeros, single-element rows, wide and tall rectangular matrices, a 200,000-nonzero row
inside a 50 x 300,000 matrix, a power-law irregular matrix, a 1x1 matrix, mixed-sign `x`, fault injection (a wrong
result must be flagged), graph versus direct timing modes, missing-input and missing-executable errors.
Not covered: matrices with zero nonzeros (the validator rejects them upstream) and CUDA compute-sanitizer runs
(see `docs/limitations.md`).
