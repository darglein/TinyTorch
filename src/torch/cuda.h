/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "tiny_torch.h"

#if __has_include("cuda_runtime.h")
#    include "cuda_runtime.h"

#ifdef __CUDACC__
#define CUDA_KERNEL_ASSERT(cond)                                         \
  if (!(cond)) {                                           \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }
#else
    #define CUDA_KERNEL_ASSERT(cond) assert(cond)
#endif
#    if defined(__CUDA_RUNTIME_H__) || defined(__CUDACC__)
namespace TINY_TORCH_NAMESPACE
{
namespace cuda
{
inline cudaStream_t getCurrentCUDAStream()
{
    throw std::runtime_error("not implemented");
    return {};
}

struct CUDACachingAllocator
{
    static void emptyCache() { throw std::runtime_error("not implemented"); }
};



}  // namespace cuda
}  // namespace TINY_TORCH_NAMESPACE
#    endif
#endif
