// Shared device code: T cooperating threads per row (T in {1,4,16,32}) and a
// block-per-row routine.  Used by CSR-vector (T=32 on all rows) and by CSR-adaptive
// (row bins).  All lanes of a warp always execute the shuffles (no early return).
#pragma once
#include "spmv_common.cuh"

// rowlist == nullptr -> row = group index; otherwise row = rowlist[group].
// `block_idx` is the (logical) block index inside this routine's own grid.
template <int T>
__device__ __forceinline__ void tpr_body(long long block_idx, int ngroups, const int* __restrict__ rowlist,
                                         const int* __restrict__ rp, const int* __restrict__ ci,
                                         const float* __restrict__ v, const float* __restrict__ x,
                                         float* __restrict__ y) {
  long long gtid = block_idx * blockDim.x + threadIdx.x;
  long long group = gtid / T;
  int lane = (int)(gtid % T);
  bool valid = group < ngroups;
  int row = valid ? (rowlist ? rowlist[group] : (int)group) : 0;
  float s = 0.f;
  if (valid) {
    int beg = rp[row], end = rp[row + 1];
    for (int k = beg + lane; k < end; k += T) s += v[k] * __ldg(&x[ci[k]]);
  }
#pragma unroll
  for (int off = T / 2; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off, T);
  if (valid && lane == 0) y[row] = s;
}

// One block (BLOCK == blockDim.x threads) per row; warp shuffle then shared-memory combine.
template <int BLOCK>
__device__ __forceinline__ void block_row_body(int row, const int* __restrict__ rp,
                                               const int* __restrict__ ci, const float* __restrict__ v,
                                               const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float part[BLOCK / 32];
  float s = 0.f;
  for (int k = rp[row] + threadIdx.x; k < rp[row + 1]; k += BLOCK) s += v[k] * __ldg(&x[ci[k]]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    float t = threadIdx.x < BLOCK / 32 ? part[threadIdx.x] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    if (threadIdx.x == 0) y[row] = t;
  }
}

template <int T>
__global__ void k_tpr(int ngroups, const int* __restrict__ rowlist, const int* __restrict__ rp,
                      const int* __restrict__ ci, const float* __restrict__ v,
                      const float* __restrict__ x, float* __restrict__ y) {
  tpr_body<T>(blockIdx.x, ngroups, rowlist, rp, ci, v, x, y);
}

template <int BLOCK>
__global__ void k_block_row(const int* __restrict__ rowlist, const int* __restrict__ rp,
                            const int* __restrict__ ci, const float* __restrict__ v,
                            const float* __restrict__ x, float* __restrict__ y) {
  block_row_body<BLOCK>(rowlist[blockIdx.x], rp, ci, v, x, y);
}

template <int T>
inline void launch_tpr(int ngroups, const int* rowlist, const DeviceCsr& d, const float* x,
                       float* y, cudaStream_t s) {
  if (ngroups <= 0) return;
  const int block = 128;
  long long threads = (long long)ngroups * T;
  long long grid = (threads + block - 1) / block;
  k_tpr<T><<<(unsigned)grid, block, 0, s>>>(ngroups, rowlist, d.rowptr, d.colidx, d.val, x, y);
  CUDA_CHECK(cudaGetLastError());
}
