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


enum class AllocatorAlgorithm
{
    CUDA_MALLOC,
    CUDA_MALLOC_ASYNC,
};
TINYTORCH_API void set_allocator_algorithm(AllocatorAlgorithm algo);


struct TINYTORCH_API CUDACachingAllocator
{
    static void emptyCache();
};



std::pair<void*,uint64_t> cuda_cached_malloc(int64_t size);
void cuda_cached_free(void* ptr, uint64_t alloc_info);

TINYTORCH_API int64_t current_allocated_size();
TINYTORCH_API int64_t max_allocated_size();

}  // namespace cuda
}  // namespace tinytorch