// Kernel B: CSR-vector.  One warp (32 lanes) per row; lanes stride over the row's
// nonzeros (consecutive lanes read consecutive val/colidx: coalesced), then a
// shuffle-down tree reduction (5 steps, no shared memory, no __syncthreads).
// Preprocessing: none.  Weakness: lanes idle on short rows (rows << 32 nnz); one warp
// per very long row limits parallelism within that row.
#include "subwarp_kernels.cuh"

class CsrVector : public SpmvKernel {
  DeviceCsr d_;

 public:
  std::string name() const override { return "csr_vector"; }
  void prepare(const HostCsr&, const DeviceCsr& d, cudaStream_t) override { d_ = d; }
  void run(const float* x, float* y, cudaStream_t s) override {
    launch_tpr<32>(d_.rows, nullptr, d_, x, y, s);
  }
  void release() override {}
};
std::unique_ptr<SpmvKernel> make_csr_vector() { return std::make_unique<CsrVector>(); }
