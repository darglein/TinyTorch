/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <mutex>
#include <set>

#include "cached_memory_allocator.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include "unary_operators.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{
namespace cuda
{

template <typename T, typename U>
constexpr T iAlignUp(T a, U b)
{
    static_assert(std::is_integral<T>::value && std::is_integral<U>::value, "only applicable to integral types");
    return (a % b != 0) ? (a - a % b + b) : a;
}

std::mutex mu;

#if 1
void* cuda_cached_malloc(int64_t size)
{
    std::unique_lock l(mu);
    void* ptr;
    cudaMallocAsync(&ptr, size, 0);
    return ptr;
}
void cuda_cached_free(void* ptr)
{
    std::unique_lock l(mu);
    cudaFreeAsync(ptr, 0);
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
    int64_t used_size;
    bool free = false;
};


static std::vector<MemoryAllocation> all_allocations;


void CheckValidMemory()
{
    for (int i = 0; i < all_allocations.size(); ++i)
    {
        for (int j = i + 1; j < all_allocations.size(); ++j)
        {
            CHECK(all_allocations[i].ptr != all_allocations[j].ptr);
        }
    }
}


void* cuda_cached_malloc(int64_t size)
{
    if (size == 0)
    {
        return nullptr;
    }

    auto padded_size = iAlignUp(size, 1024 * 1024);
    std::unique_lock l(mu);
    CheckValidMemory();
    for (auto& a : all_allocations)
    {
        if (a.size > padded_size + padded_size / 2)
        {
            // the cached chunk is larger than 50% more of tensor size
            // this is too much wasted memory -> skip
            continue;
        }

        if (a.size >= size && a.free)
        {
            a.free      = false;
            a.used_size = size;
            return a.ptr;
        }
    }

    MemoryAllocation new_alloc;

    // use a minimum allocation size and alignment
    new_alloc.size = padded_size;

    cudaError_t malloc_error;
    while (true)
    {
        malloc_error = cudaMalloc(&new_alloc.ptr, new_alloc.size);

        if (malloc_error == cudaErrorMemoryAllocation)
        {
            // free the largest unused memory chunk
            int freed_alloc = -1;
            for (int i = all_allocations.size() - 1; i >= 0; --i)
            {
                if (all_allocations[i].free)
                {
                    CHECK_CUDA_ERROR(cudaFree(all_allocations[i].ptr));
                    freed_alloc = i;
                    break;
                }
            }

            if (freed_alloc == -1)
            {
                CHECK(false) << "out of memory!";
            }
            else
            {
                all_allocations.erase(all_allocations.begin() + freed_alloc);
                continue;
            }
        }
        CHECK_EQ(malloc_error, 0);
        break;
    }

    new_alloc.free = false;
    all_allocations.push_back(new_alloc);

    // sort by size (smallest size first) -> we get best fit allocation
    std::sort(all_allocations.begin(), all_allocations.end(), [](auto a, auto b) { return a.size < b.size; });

    CheckValidMemory();

    return new_alloc.ptr;
}

void cuda_cached_free(void* ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    std::unique_lock l(mu);
    CheckValidMemory();
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
    for (auto& a : all_allocations)
    {
        if (a.free)
        {
            CHECK_CUDA_ERROR(cudaFree(a.ptr));
        }
    }
    all_allocations.erase(std::remove_if(all_allocations.begin(), all_allocations.end(), [](auto a) { return a.free; }),
                          all_allocations.end());
    CheckValidMemory();
}

#endif
}  // namespace cuda

}  // namespace tinytorch