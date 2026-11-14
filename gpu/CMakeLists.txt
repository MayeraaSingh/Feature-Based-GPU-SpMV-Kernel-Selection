cmake_minimum_required(VERSION 3.24)
project(spmv_gpu LANGUAGES CXX CUDA)

# Build for the architecture of the machine this is compiled on ("native"); fall back to
# 86 (RTX 3050 laptop, the project's reference GPU) if native detection is unavailable.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(spmv_kernels STATIC
  kernels/csr_scalar.cu
  kernels/csr_vector.cu
  kernels/csr_adaptive.cu
  kernels/cusparse_baseline.cu
  kernels/row_stats.cu)
target_include_directories(spmv_kernels PUBLIC kernels)
target_link_libraries(spmv_kernels PUBLIC CUDA::cudart CUDA::cusparse)

add_executable(spmv_bench benchmarks/benchmark_runner.cu)
target_link_libraries(spmv_bench PRIVATE spmv_kernels)
