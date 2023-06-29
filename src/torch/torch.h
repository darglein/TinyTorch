/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#define TINY_TORCH_NAMESPACE torch
#include "cuda_runtime.h"
#include "tiny_torch.h"

#if defined(__CUDA_RUNTIME_H__) || defined(__CUDACC__)
namespace TINY_TORCH_NAMESPACE
{
namespace cuda
{
inline cudaStream_t getCurrentCUDAStream()
{
    throw std::runtime_error("not implemented");
    return {};
}
}  // namespace cuda
}  // namespace TINY_TORCH_NAMESPACE
#endif


// namespace torch=TINY_TORCH_NAMESPACE;
namespace at = TINY_TORCH_NAMESPACE;