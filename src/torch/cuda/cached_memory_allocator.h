/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/tiny_torch_config.h"
namespace tinytorch
{
namespace cuda
{

struct TINYTORCH_API CUDACachingAllocator
{
    static void emptyCache();
};


void* cuda_cached_malloc(int64_t size);
void cuda_cached_free(void* ptr);

}
}  // namespace tinytorch