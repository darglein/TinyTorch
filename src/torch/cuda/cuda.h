/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/tensor.h"

#include "cuda_runtime.h"
#include "cached_memory_allocator.h"

#ifdef __CUDACC__
#    define CUDA_KERNEL_ASSERT(cond) \
        if (!(cond))                 \
        {                            \
            assert(cond);            \
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
