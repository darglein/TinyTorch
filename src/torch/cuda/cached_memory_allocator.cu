/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <set>

#include "cached_memory_allocator.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include "unary_operators.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mutex>


namespace tinytorch
{
namespace cuda
{

std::mutex mu;

#if 0
void* cuda_cached_malloc(int64_t size)
{
    std::unique_lock l(mu);
    void* ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}
void cuda_cached_free(void* ptr)
{
    std::unique_lock l(mu);
    cudaFree(ptr);
    return;
}
void CUDACachingAllocator::emptyCache()
{
}

#else
struct MemoryAllocation
{
    void* ptr;
    int64_t size;
    bool free = false;
};


static std::vector<MemoryAllocation> all_allocations;



void* cuda_cached_malloc(int64_t size)
{
    std::unique_lock l(mu);
    for (auto& a : all_allocations)
    {
        if (a.size >= size && a.free)
        {
            a.free = false;
            return a.ptr;
        }
    }

    MemoryAllocation new_alloc;
    new_alloc.size = size;
    CHECK_CUDA_ERROR(cudaMalloc(&new_alloc.ptr, size));
    new_alloc.free = false;
    all_allocations.push_back(new_alloc);
    return new_alloc.ptr;
}

void cuda_cached_free(void* ptr)
{
    std::unique_lock l(mu);
    for (auto& a : all_allocations)
    {
        if (a.ptr == ptr)
        {
            CHECK(!a.free);
            a.free = true;
            return;
        }
    }
    CHECK(false);
}
void CUDACachingAllocator::emptyCache()
{
    std::unique_lock l(mu);
    // all_allocations.erase(std::remove_if(all_allocations.begin(), all_allocations.end(), [](auto a) { return a.free; }),
    //                       all_allocations.end());
}

#endif
}  // namespace cuda

}  // namespace tinytorch