# GPU Module

**Owner:** Member 1 (GPU kernels and GPU benchmarking)

## Purpose

The `gpu` module contains CUDA/C++ implementations of CSR SpMV kernels and CUDA benchmark harnesses.

## Subdirectories

- `kernels/`: CUDA kernel implementations (e.g., CSR-scalar, CSR-vector, CSR-adaptive, cuSPARSE baseline wrappers).
- `benchmarks/`: Standalone benchmarking executables, timing harnesses, and correctness verification routines against the CPU reference.

*Note: Work in this directory requires the CUDA-capable environment.*

