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
    UNKNOWN = 0,
    CUDA_MALLOC,
    CUDA_MALLOC_ASYNC,
    CUDA_MALLOC_CHECK_FREE,
    CUDA_PRE_ALLOCATE,
};
TINYTORCH_API void set_allocator_algorithm(AllocatorAlgorithm algo);
TINYTORCH_API AllocatorAlgorithm get_allocator_algorithm();

// 0: on_error
// 1: on_out_of_memory
// 2: info
TINYTORCH_API void set_allocator_log_level(int level);
TINYTORCH_API int get_allocator_log_level();

// If needed is larger, than it can fit into VRAM, less is allocated and the amount of actual allocate VRAM is returned.
TINYTORCH_API int64_t pre_allocate_vram(int64_t needed);
TINYTORCH_API void  free_preallocate_vram();
TINYTORCH_API int64_t prealloc_free_list_size();
TINYTORCH_API int64_t prealloc_free_memory();


struct TINYTORCH_API CUDACachingAllocator
{
    static void emptyCache();
};



std::pair<void*,uint64_t> cuda_cached_malloc(int64_t size, int device_id);
void cuda_cached_free(void* ptr, uint64_t alloc_info, int device_id);

TINYTORCH_API int64_t current_allocated_size(int device_id);
TINYTORCH_API int64_t max_allocated_size(int device_id);


void* cuda_malloc_pinned(int64_t size);
void cuda_pinned_free(void* ptr);

}  // namespace cuda
}  // namespace tinytorch