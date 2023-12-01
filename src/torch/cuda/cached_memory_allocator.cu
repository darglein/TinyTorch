/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <map>
#include <mutex>

#include "cached_memory_allocator.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>


namespace tinytorch
{
namespace cuda
{

std::mutex mu;
static std::map<void*, int64_t> allocated_blocks;
static bool debug_print         = false;
static int64_t debug_print_size = 1000 * 1000 * 300;

static int64_t current_allocated_bytes = 0;
static int64_t max_allocated_bytes     = 0;

int64_t current_allocated_size()
{
    return current_allocated_bytes;
}
int64_t max_allocated_size()
{
    return max_allocated_bytes;
}


static void* malloc_async(int64_t size)
{
    if (size == 0)
    {
        return nullptr;
    }
    std::unique_lock l(mu);
    void* ptr;
    auto strm       = cuda::getCurrentCUDAStream();
    auto cuda_error = cudaMallocAsync(&ptr, size, strm);
    if (cuda_error == cudaErrorMemoryAllocation)
    {
        size_t mem_free, mem_total;
        cudaMemGetInfo(&mem_free, &mem_total);
        CHECK_NE(cuda_error, cudaErrorMemoryAllocation)
            << " CUDA out of memory!\n"
            << "     Tried to allocate " << (size / 1000.0 / 1000.0) << "MB\n"
            << "     Free memory " << (mem_free / 1000.0 / 1000.0) << "MB\n"
            << "     Total memory " << (mem_total / 1000.0 / 1000.0) << "MB\n"
            << "     Allocated by torch " << (current_allocated_bytes / 1000.0 / 1000.0) << "MB\n";
    }
    CHECK_CUDA_ERROR(cuda_error);
    CHECK_NOTNULL(ptr);

    CHECK(allocated_blocks.find(ptr) == allocated_blocks.end());
    allocated_blocks.insert({ptr, size});
    current_allocated_bytes += size;
    max_allocated_bytes = std::max(current_allocated_bytes, max_allocated_bytes);
    CHECK(allocated_blocks.find(ptr) != allocated_blocks.end());

    if (debug_print)
    {
        if (size > debug_print_size)
        {
            std::cout << "Allocate CUDA Memory: " << (size / 1000.0 / 1000.0) << "MB (" << ptr
                      << ") Curr. Alloc: " << (current_allocated_bytes / (1000.0 * 1000.0)) << " MB\n";
        }
    }

    return ptr;
}

static void free_async(void* ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    std::unique_lock l(mu);
    CHECK(allocated_blocks.find(ptr) != allocated_blocks.end());
    int64_t size = allocated_blocks.find(ptr)->second;

    if (debug_print)
    {
        if (size > debug_print_size)
        {
            std::cout << "Free CUDA Memory: " << (size / 1000.0 / 1000.0) << "MB (" << ptr << ")"
                      << "\n";
        }
    }

    current_allocated_bytes -= size;
    allocated_blocks.erase(ptr);

    cudaFreeAsync(ptr, cuda::getCurrentCUDAStream());
}

void* cuda_cached_malloc(int64_t size)
{
    auto ptr = malloc_async(size);

    return ptr;
}
void cuda_cached_free(void* ptr)
{
    free_async(ptr);
}

void CUDACachingAllocator::emptyCache()
{
    // this frees unused values for the async allocator
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

}  // namespace cuda

}  // namespace tinytorch