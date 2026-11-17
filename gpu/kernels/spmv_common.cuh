// Common types and the kernel interface shared by all CSR SpMV candidates.
// All candidates consume the same CSR matrix (int32 indices, float32 values),
// the same x and write the same y (float32).  y = A * x.
#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                \
  do {                                                                                  \
    cudaError_t err_ = (call);                                                          \
    if (err_ != cudaSuccess) {                                                          \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__) +     \
                               " in " #call);                                           \
    }                                                                                   \
  } while (0)

struct HostCsr {
  int rows = 0, cols = 0, nnz = 0;
  std::vector<int> rowptr, colidx;
  std::vector<float> val;
};

struct DeviceCsr {
  int rows = 0, cols = 0, nnz = 0;
  int* rowptr = nullptr;
  int* colidx = nullptr;
  float* val = nullptr;
};

// Interface: prepare() does any per-matrix preprocessing (timed separately by the
// harness); run() enqueues the SpMV on `stream` (no synchronisation, no allocation,
// so it can be captured into a CUDA graph); release() frees what prepare() allocated.
class SpmvKernel {
 public:
  virtual ~SpmvKernel() {}
  virtual std::string name() const = 0;
  virtual void prepare(const HostCsr& h, const DeviceCsr& d, cudaStream_t stream) = 0;
  // Called once, untimed, before graph capture, with the final x and y buffers.
  virtual void bind(const float* x, float* y, cudaStream_t stream) {}
  virtual void run(const float* x, float* y, cudaStream_t stream) = 0;
  virtual void release() = 0;
};

std::unique_ptr<SpmvKernel> make_csr_scalar();
std::unique_ptr<SpmvKernel> make_csr_vector();
std::unique_ptr<SpmvKernel> make_csr_adaptive();            // fused single-launch (candidate)
std::unique_ptr<SpmvKernel> make_csr_adaptive_multilaunch(); // design iteration 1 (documentation only)
std::unique_ptr<SpmvKernel> make_cusparse();

// Row-length statistics gathered on the device (feature-gathering overhead study).
struct RowStats {
  unsigned long long sumsq;
  int maxlen, minlen;
};
void gpu_row_stats(const DeviceCsr& d, RowStats* out_host, cudaStream_t stream);
