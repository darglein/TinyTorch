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

static constexpr int MAX_DEVICES = 8;

static std::mutex mu;
static bool allocator_initialized = false;
static bool has_malloc_async      = false;
static bool debug_print           = false;
static int64_t debug_print_size   = 25 * 1024 * 1024;

struct PerDeviceMemoryData
{
    std::map<void*, int64_t> allocated_blocks;

    int64_t current_allocated_bytes = 0;
    int64_t max_allocated_bytes     = 0;
};

static PerDeviceMemoryData& DeviceData(int device_id)
{
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, MAX_DEVICES);
    static PerDeviceMemoryData data[MAX_DEVICES];
    return data[device_id];
}


static thread_local AllocatorAlgorithm algorithm = AllocatorAlgorithm::CUDA_MALLOC_ASYNC;


void set_allocator_algorithm(AllocatorAlgorithm algo)
{
    std::unique_lock l(mu);
    algorithm = algo;
}

int64_t current_allocated_size(int device_id)
{
    return DeviceData(device_id).current_allocated_bytes;
}
int64_t max_allocated_size(int device_id)
{
    return DeviceData(device_id).max_allocated_bytes;
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

static void handle_cuda_allocation_error(cudaError_t cuda_error, int64_t size, int device_id)
{
    if (cuda_error == cudaErrorMemoryAllocation)
    {
        size_t mem_free, mem_total;
        cudaMemGetInfo(&mem_free, &mem_total);
        std::cerr << " CUDA out of memory!\n"
                  << "     Tried to allocate " << (size / 1000.0 / 1000.0) << "MB\n"
                  << "     Free memory " << (mem_free / 1000.0 / 1000.0) << "MB\n"
                  << "     Total memory " << (mem_total / 1000.0 / 1000.0) << "MB\n"
                  << "     Allocated by torch " << (DeviceData(device_id).current_allocated_bytes / 1000.0 / 1000.0)
                  << "MB\n";

        throw std::runtime_error(std::string("CUDA memory allocation error: ") + cudaGetErrorString(cuda_error));
    }
}

static void* malloc_async(int64_t size, int device_id)
{
    void* ptr;
    auto strm       = cuda::getCurrentCUDAStream();
    auto cuda_error = cudaMallocAsync(&ptr, size, strm);

    handle_cuda_allocation_error(cuda_error, size, device_id);

    CHECK_CUDA_ERROR(cuda_error);
    CHECK_NOTNULL(ptr);

    return ptr;
}

static void free_async(void* ptr)
{
    CHECK_CUDA_ERROR(cudaFreeAsync(ptr, cuda::getCurrentCUDAStream()));
}

static void* malloc_blocking(int64_t size, int device_id)
{
    void* ptr;
    cudaError_t cuda_error = cudaMalloc(&ptr, size);

    handle_cuda_allocation_error(cuda_error, size, device_id);

    return ptr;
}

static void free_blocking(void* ptr)
{
    // not quite sure why the cuda device synchronize is needed
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

std::pair<void*, uint64_t> cuda_cached_malloc(int64_t size, int device_id)
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
        ptr  = malloc_async(size, device_id);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC_ASYNC;
    }

    if (!ptr)
    {
        ptr  = malloc_blocking(size, device_id);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC;
    }


    if (ptr)
    {
        auto& d = DeviceData(device_id);
        CHECK(d.allocated_blocks.find(ptr) == d.allocated_blocks.end());
        d.allocated_blocks.insert({ptr, size});
        d.current_allocated_bytes += size;
        d.max_allocated_bytes = std::max(d.current_allocated_bytes, d.max_allocated_bytes);
        CHECK(d.allocated_blocks.find(ptr) != d.allocated_blocks.end());

        if (debug_print)
        {
            if (size > debug_print_size)
            {
                std::cout << "Allocate CUDA Memory with algo= " << info << " on device " << getDevice()
                          << " size: " << (size / 1024.0 / 1024.0) << "MiB (" << ptr
                          << ") Curr. Alloc: " << (d.current_allocated_bytes / (1024.0 * 1024.0)) << " MiB\n";
            }
        }
    }

    return {ptr, info};
}
void cuda_cached_free(void* ptr, uint64_t alloc_info, int device_id)
{
    if (ptr == nullptr)
    {
        return;
    }

    auto algo = (AllocatorAlgorithm)alloc_info;
    CHECK(allocator_initialized);

    std::unique_lock l(mu);


    {
        auto& d = DeviceData(device_id);
        CHECK(d.allocated_blocks.find(ptr) != d.allocated_blocks.end());
        int64_t size = d.allocated_blocks.find(ptr)->second;
        if (debug_print)
        {
            if (size > debug_print_size)
            {
                std::cout << "Free CUDA Memory with algo= " << (int)algo << " size " << (size / 1024.0 / 1024.0)
                          << "MiB (" << ptr << ")" << "\n";
            }
        }

        d.current_allocated_bytes -= size;
        d.allocated_blocks.erase(ptr);
    }



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