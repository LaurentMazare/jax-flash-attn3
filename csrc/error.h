#pragma once

#define C10_CUDA_CHECK(EXPR)                                        \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    if (__err != cudaSuccess) {                                     \
        std::cerr << "CUDA error " << cudaGetErrorString(__err) << std::endl; \
    }                                                               \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
