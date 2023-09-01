/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/tensor.h"

#include "cached_memory_allocator.h"
#include "cuda_runtime.h"

#ifdef __CUDACC__

#    undef assert
inline __device__ void __assert_fail_cuda(const char* __assertion, const char* __file, unsigned int __line,
                                          const char* __function)
{
    printf("cuda assert failed in %s:%d func: %s\n block:     %d %d %d\n thread:    %d %d %d\n Assertion: %s\n", __file,
           __line, __function, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, __assertion);
    __trap();
}

#    define CUDA_KERNEL_ASSERT(cond)                                                            \
        if ((!(cond)))                                                                          \
        {                                                                                       \
            __assert_fail_cuda(#cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
        }

#else
#    define CUDA_KERNEL_ASSERT(cond) assert(cond)
#endif
#if defined(__CUDA_RUNTIME_H__) || defined(__CUDACC__)
#endif

namespace tinytorch
{
namespace cuda
{
TINYTORCH_API cudaStream_t getCurrentCUDAStream();

TINYTORCH_API void setCUDAStreamForThisThread(cudaStream_t stream);

}  // namespace cuda
}  // namespace tinytorch
