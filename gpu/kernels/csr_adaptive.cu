// Kernel C: CSR-adaptive (row-length binning).
// Preprocessing (separately timed): the host scans rowptr, classifies every row by its
// length L and uploads one row-index list per bin:
//   bin 0: L <= 4     -> 1 thread / row               (<= 4 nnz per thread)
//   bin 1: L <= 16    -> 4 threads / row              (<= 4 nnz per thread)
//   bin 2: L <= 64    -> 16 threads / row             (<= 4 nnz per thread)
//   bin 3: L <= 512   -> 32 threads (warp) / row      (<= 16 nnz per lane)
//   bin 4: L  > 512   -> 128-thread block / row
// The thresholds are FIXED design constants derived from "about <= 4 nonzeros per thread"
// and the warp/block sizes.  They were NOT tuned on timings and are not claimed to be
// optimal.  Empty rows land in bin 0 (they write 0).  No atomics.
//
// Two implementations of the same strategy:
//   * csr_adaptive            ONE fused launch: the grid is the concatenation of the bins'
//                             blocks and every block dispatches on its bin (block-uniform
//                             branch, so no divergence).  Design iteration 2.
//   * csr_adaptive_multilaunch one launch per non-empty bin (design iteration 1; kept for
//                             the documented comparison, not a candidate in the experiment).
// The preprocessing depends only on the sparsity pattern, so it can be amortised across SpMV
// calls only if the pattern is reused.
#include "subwarp_kernels.cuh"

static const int kBlock = 128;
static const int kTpr[4] = {1, 4, 16, 32};

static int bin_of(int len) {
  if (len <= 4) return 0;
  if (len <= 16) return 1;
  if (len <= 64) return 2;
  if (len <= 512) return 3;
  return 4;
}

struct AdaptiveParams {
  const int* list[5];
  int count[5];
  int block_start[6];  // cumulative block offsets of the bins; block_start[5] = total blocks
};

__global__ void k_adaptive_fused(const __grid_constant__ AdaptiveParams P, const int* __restrict__ rp, const int* __restrict__ ci,
                                 const float* __restrict__ v, const float* __restrict__ x,
                                 float* __restrict__ y) {
  const int bid = blockIdx.x;
  int b = 0;
#pragma unroll
  for (int i = 1; i < 5; ++i) b = (bid >= P.block_start[i]) ? i : b;  // largest i with start <= bid
  const int lb = bid - P.block_start[b];
  switch (b) {
    case 0: tpr_body<1>(lb, P.count[0], P.list[0], rp, ci, v, x, y); break;
    case 1: tpr_body<4>(lb, P.count[1], P.list[1], rp, ci, v, x, y); break;
    case 2: tpr_body<16>(lb, P.count[2], P.list[2], rp, ci, v, x, y); break;
    case 3: tpr_body<32>(lb, P.count[3], P.list[3], rp, ci, v, x, y); break;
    default: block_row_body<kBlock>(P.list[4][lb], rp, ci, v, x, y); break;
  }
}

class CsrAdaptiveBase : public SpmvKernel {
 protected:
  DeviceCsr d_;
  int* list_[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  int count_[5] = {0, 0, 0, 0, 0};

  void build(const HostCsr& h, const DeviceCsr& d) {
    d_ = d;
    std::vector<int> lists[5];
    for (int r = 0; r < h.rows; ++r) lists[bin_of(h.rowptr[r + 1] - h.rowptr[r])].push_back(r);
    for (int b = 0; b < 5; ++b) {
      count_[b] = (int)lists[b].size();
      if (count_[b] > 0) {
        CUDA_CHECK(cudaMalloc(&list_[b], count_[b] * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(list_[b], lists[b].data(), count_[b] * sizeof(int), cudaMemcpyHostToDevice));
      }
    }
  }

 public:
  ~CsrAdaptiveBase() override { release(); }
  void release() override {
    for (int b = 0; b < 5; ++b) {
      if (list_[b]) cudaFree(list_[b]);
      list_[b] = nullptr;
      count_[b] = 0;
    }
  }
};

class CsrAdaptive : public CsrAdaptiveBase {
  AdaptiveParams P_{};

 public:
  std::string name() const override { return "csr_adaptive"; }
  void prepare(const HostCsr& h, const DeviceCsr& d, cudaStream_t) override {
    build(h, d);
    long long start = 0;
    for (int b = 0; b < 5; ++b) {
      P_.list[b] = list_[b];
      P_.count[b] = count_[b];
      P_.block_start[b] = (int)start;
      long long blocks = (b < 4) ? ((long long)count_[b] * kTpr[b] + kBlock - 1) / kBlock : count_[b];
      start += blocks;
    }
    P_.block_start[5] = (int)start;
  }
  void run(const float* x, float* y, cudaStream_t s) override {
    if (P_.block_start[5] == 0) return;
    k_adaptive_fused<<<(unsigned)P_.block_start[5], kBlock, 0, s>>>(P_, d_.rowptr, d_.colidx, d_.val, x, y);
    CUDA_CHECK(cudaGetLastError());
  }
};

class CsrAdaptiveMultiLaunch : public CsrAdaptiveBase {
 public:
  std::string name() const override { return "csr_adaptive_multilaunch"; }
  void prepare(const HostCsr& h, const DeviceCsr& d, cudaStream_t) override { build(h, d); }
  void run(const float* x, float* y, cudaStream_t s) override {
    launch_tpr<1>(count_[0], list_[0], d_, x, y, s);
    launch_tpr<4>(count_[1], list_[1], d_, x, y, s);
    launch_tpr<16>(count_[2], list_[2], d_, x, y, s);
    launch_tpr<32>(count_[3], list_[3], d_, x, y, s);
    if (count_[4] > 0) {
      k_block_row<kBlock><<<count_[4], kBlock, 0, s>>>(list_[4], d_.rowptr, d_.colidx, d_.val, x, y);
      CUDA_CHECK(cudaGetLastError());
    }
  }
};

std::unique_ptr<SpmvKernel> make_csr_adaptive() { return std::make_unique<CsrAdaptive>(); }
std::unique_ptr<SpmvKernel> make_csr_adaptive_multilaunch() { return std::make_unique<CsrAdaptiveMultiLaunch>(); }
