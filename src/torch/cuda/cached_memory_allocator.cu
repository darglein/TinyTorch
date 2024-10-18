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


static bool allocator_initialized = false;
static bool has_malloc_async      = false;

static AllocatorAlgorithm algorithm = AllocatorAlgorithm::CUDA_MALLOC_ASYNC;


void set_allocator_algorithm(AllocatorAlgorithm algo)
{
    std::unique_lock l(mu);
    algorithm = algo;
}

int64_t current_allocated_size()
{
    return current_allocated_bytes;
}
int64_t max_allocated_size()
{
    return max_allocated_bytes;
}

static void initialize_allocator()
{
    CHECK(!allocator_initialized);
    allocator_initialized = true;

    int value = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, 0));

    if (value == 0)
    {
        has_malloc_async = false;
    }
    else
    {
        has_malloc_async = true;
    }
}

static void handle_cuda_allocation_error(cudaError_t cuda_error, int64_t size)
{
    if (cuda_error == cudaErrorMemoryAllocation)
    {
        size_t mem_free, mem_total;
        cudaMemGetInfo(&mem_free, &mem_total);
        std::cerr << " CUDA out of memory!\n"
                  << "     Tried to allocate " << (size / 1000.0 / 1000.0) << "MB\n"
                  << "     Free memory " << (mem_free / 1000.0 / 1000.0) << "MB\n"
                  << "     Total memory " << (mem_total / 1000.0 / 1000.0) << "MB\n"
                  << "     Allocated by torch " << (current_allocated_bytes / 1000.0 / 1000.0) << "MB\n";

        throw std::runtime_error(std::string("CUDA memory allocation error: ") + cudaGetErrorString(cuda_error));
    }
}

static void* malloc_async(int64_t size)
{
    void* ptr;
    auto strm       = cuda::getCurrentCUDAStream();
    auto cuda_error = cudaMallocAsync(&ptr, size, strm);

    handle_cuda_allocation_error(cuda_error, size);

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
    CHECK(allocated_blocks.find(ptr) != allocated_blocks.end());
    int64_t size = allocated_blocks.find(ptr)->second;

    if (debug_print)
    {
        if (size > debug_print_size)
        {
            std::cout << "Free CUDA Memory: " << (size / 1000.0 / 1000.0) << "MB (" << ptr << ")" << "\n";
        }
    }

    current_allocated_bytes -= size;
    allocated_blocks.erase(ptr);

    cudaFreeAsync(ptr, cuda::getCurrentCUDAStream());
}

static void* malloc_blocking(int64_t size)
{
    void* ptr;
    cudaError_t cuda_error = cudaMalloc(&ptr, size);

    handle_cuda_allocation_error(cuda_error, size);

    return ptr;
}

static void free_blocking(void* ptr)
{
    cudaFree(ptr);
}

std::pair<void*, uint64_t> cuda_cached_malloc(int64_t size)
{
    if (size == 0)
    {
        return {nullptr, 0};
    }
    std::unique_lock l(mu);

    if (!allocator_initialized)
    {
        initialize_allocator();
    }

    void* ptr = nullptr;
    uint64_t info;

    if (has_malloc_async && algorithm == AllocatorAlgorithm::CUDA_MALLOC_ASYNC)
    {
        ptr  = malloc_async(size);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC_ASYNC;
    }

    if (!ptr)
    {
        ptr  = malloc_blocking(size);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC;
    }
    return {ptr, info};
}
void cuda_cached_free(void* ptr, uint64_t alloc_info)
{
    if (ptr == nullptr)
    {
        return;
    }

    auto algo = (AllocatorAlgorithm)alloc_info;
    CHECK(allocator_initialized);

    std::unique_lock l(mu);

    if (algo == AllocatorAlgorithm::CUDA_MALLOC_ASYNC)
    {
        free_async(ptr);
    }
    else if (algo == AllocatorAlgorithm::CUDA_MALLOC)
    {
        free_blocking(ptr);
    }
    else
    {
        CHECK(false);
    }
}

void CUDACachingAllocator::emptyCache()
{
    CHECK(allocator_initialized);
    // this frees unused values for the async allocator
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

}  // namespace cuda

}  // namespace tinytorch