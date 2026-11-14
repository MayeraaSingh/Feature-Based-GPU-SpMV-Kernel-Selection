// Kernel A: CSR-scalar.  One thread per row (grid-stride over rows).
// Memory: the threads of a warp read from 32 different rows (uncoalesced on val/colidx
// unless rows are very short); x is gathered through the read-only path.
// Reduction: none (sequential dot product in one thread).  Sync: none.
// Preprocessing: none.  Weakness: load imbalance / divergence for irregular rows,
// long rows serialise on one thread.
#include "spmv_common.cuh"

__global__ void k_csr_scalar(int rows, const int* __restrict__ rp, const int* __restrict__ ci,
                             const float* __restrict__ v, const float* __restrict__ x,
                             float* __restrict__ y) {
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
    float s = 0.f;
    for (int k = rp[r]; k < rp[r + 1]; ++k) s += v[k] * __ldg(&x[ci[k]]);
    y[r] = s;
  }
}

class CsrScalar : public SpmvKernel {
  DeviceCsr d_;

 public:
  std::string name() const override { return "csr_scalar"; }
  void prepare(const HostCsr&, const DeviceCsr& d, cudaStream_t) override { d_ = d; }
  void run(const float* x, float* y, cudaStream_t s) override {
    if (d_.rows == 0) return;
    const int block = 256;
    long long grid = ((long long)d_.rows + block - 1) / block;
    k_csr_scalar<<<(unsigned)grid, block, 0, s>>>(d_.rows, d_.rowptr, d_.colidx, d_.val, x, y);
    CUDA_CHECK(cudaGetLastError());
  }
  void release() override {}
};
std::unique_ptr<SpmvKernel> make_csr_scalar() { return std::make_unique<CsrScalar>(); }
