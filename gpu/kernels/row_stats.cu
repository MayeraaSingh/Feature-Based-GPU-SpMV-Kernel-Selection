// Device-side row-length statistics (sum of squares, max, min) from rowptr.  Used only
// to MEASURE the cost of gathering the row-distribution features on the GPU.
#include "spmv_common.cuh"

__global__ void k_row_stats(int rows, const int* __restrict__ rp, unsigned long long* sumsq,
                            int* mx, int* mn) {
  unsigned long long sq = 0;
  int lmax = 0, lmin = 0x7fffffff;
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
    int len = rp[r + 1] - rp[r];
    sq += (unsigned long long)len * (unsigned long long)len;
    lmax = max(lmax, len);
    lmin = min(lmin, len);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_down_sync(0xffffffffu, sq, off);
    lmax = max(lmax, __shfl_down_sync(0xffffffffu, lmax, off));
    lmin = min(lmin, __shfl_down_sync(0xffffffffu, lmin, off));
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(sumsq, sq);
    atomicMax(mx, lmax);
    atomicMin(mn, lmin);
  }
}

void gpu_row_stats(const DeviceCsr& d, RowStats* out, cudaStream_t s) {
  static unsigned long long* d_sq = nullptr;
  static int* d_m = nullptr;  // [0]=max [1]=min
  if (!d_sq) {
    CUDA_CHECK(cudaMalloc(&d_sq, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_m, 2 * sizeof(int)));
  }
  int init[2] = {0, 0x7fffffff};
  CUDA_CHECK(cudaMemsetAsync(d_sq, 0, sizeof(unsigned long long), s));
  CUDA_CHECK(cudaMemcpyAsync(d_m, init, sizeof(init), cudaMemcpyHostToDevice, s));
  int grid = (int)(((long long)d.rows + 255) / 256);
  if (grid > 1024) grid = 1024;
  if (grid < 1) grid = 1;
  k_row_stats<<<grid, 256, 0, s>>>(d.rows, d.rowptr, d_sq, d_m, d_m + 1);
  CUDA_CHECK(cudaGetLastError());
  int hm[2];
  CUDA_CHECK(cudaMemcpyAsync(&out->sumsq, d_sq, sizeof(unsigned long long), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(hm, d_m, sizeof(hm), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  out->maxlen = hm[0];
  out->minlen = hm[1];
}
