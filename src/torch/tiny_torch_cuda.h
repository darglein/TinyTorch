/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#if __has_include("cuda_runtime.h")

#    include "cuda_runtime.h"

#include "torch/cuda/cached_memory_allocator.h"

#    ifdef __CUDACC__
#        define CUDA_KERNEL_ASSERT(cond) \
            if (!(cond))                 \
            {                            \
                assert(cond);             \
            }
#    else
#        define CUDA_KERNEL_ASSERT(cond) assert(cond)
#    endif
#    if defined(__CUDA_RUNTIME_H__) || defined(__CUDACC__)
namespace tinytorch
{
namespace cuda
{
inline cudaStream_t getCurrentCUDAStream()
{
    return 0;
}

}  // namespace cuda
}  // namespace tinytorch
#    endif

#endif
