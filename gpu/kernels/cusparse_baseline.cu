// Kernel D: NVIDIA cuSPARSE vendor baseline (NOT a custom kernel).
// cusparseSpMV, CSR, int32 indices, float32 values and compute type, ALG_DEFAULT,
// op = non-transpose.  Preprocessing (timed): handle + descriptor creation, workspace
// size query, workspace allocation, cusparseSpMV_preprocess.  The timed run() is the
// cusparseSpMV call only.
#include <cusparse.h>

#include "spmv_common.cuh"

#define CUSPARSE_CHECK(call)                                                                 \
  do {                                                                                       \
    cusparseStatus_t st_ = (call);                                                           \
    if (st_ != CUSPARSE_STATUS_SUCCESS)                                                      \
      throw std::runtime_error(std::string("cuSPARSE error: ") + cusparseGetErrorString(st_) + \
                               " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
  } while (0)

class CuSparseSpmv : public SpmvKernel {
  cusparseHandle_t h_ = nullptr;
  cusparseSpMatDescr_t A_ = nullptr;
  cusparseDnVecDescr_t X_ = nullptr, Y_ = nullptr;
  void* buf_ = nullptr;
  DeviceCsr d_;
  const float alpha_ = 1.f, beta_ = 0.f;

  void destroy_descriptors() {
    if (A_) { cusparseDestroySpMat(A_); A_ = nullptr; }
    if (X_) { cusparseDestroyDnVec(X_); X_ = nullptr; }
    if (Y_) { cusparseDestroyDnVec(Y_); Y_ = nullptr; }
    if (buf_) { cudaFree(buf_); buf_ = nullptr; }
  }

 public:
  ~CuSparseSpmv() override {
    release();
    if (h_) { cusparseDestroy(h_); h_ = nullptr; }
  }
  std::string name() const override { return "cusparse"; }
  void prepare(const HostCsr&, const DeviceCsr& d, cudaStream_t s) override {
    d_ = d;
    if (!h_) CUSPARSE_CHECK(cusparseCreate(&h_));
    CUSPARSE_CHECK(cusparseSetStream(h_, s));
  }
  // Descriptors need the final x / y pointers, so the harness calls bind() once.
  // bind() is part of preprocessing for cuSPARSE and is included in its timing.
  void bind(const float* x, float* y, cudaStream_t) override {
    destroy_descriptors();
    CUSPARSE_CHECK(cusparseCreateCsr(&A_, d_.rows, d_.cols, d_.nnz, d_.rowptr, d_.colidx, d_.val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&X_, d_.cols, (void*)x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&Y_, d_.rows, (void*)y, CUDA_R_32F));
    size_t bs = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(h_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, A_, X_,
                                           &beta_, Y_, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bs));
    CUDA_CHECK(cudaMalloc(&buf_, bs ? bs : 1));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(h_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, A_, X_,
                                           &beta_, Y_, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           buf_));
  }
  void run(const float*, float*, cudaStream_t) override {
    CUSPARSE_CHECK(cusparseSpMV(h_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, A_, X_, &beta_, Y_,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf_));
  }
  // The handle is created once per process (as an application would) and is not part
  // of the per-matrix preprocessing; descriptors + workspace are released per matrix.
  void release() override { destroy_descriptors(); }
};
std::unique_ptr<SpmvKernel> make_cusparse() { return std::make_unique<CuSparseSpmv>(); }
