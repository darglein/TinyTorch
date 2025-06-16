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
static int log_level                             = 1;
const int64_t log_level_size_th                  = 10 * 1024 * 1024;


void set_allocator_algorithm(AllocatorAlgorithm algo)
{
    std::unique_lock l(mu);
    algorithm = algo;
}

AllocatorAlgorithm get_allocator_algorithm()
{
    return algorithm;
}

void set_allocator_log_level(int level)
{
    std::unique_lock l(mu);
    log_level = level;
}

int get_allocator_log_level()
{
    std::unique_lock l(mu);
    return log_level;
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
    TT_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, 0));

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
        if (log_level >= 1)
        {
            size_t mem_free, mem_total;
            cudaMemGetInfo(&mem_free, &mem_total);
            std::cout << " CUDA out of memory!\n"
                      << "     Tried to allocate " << (size / 1024.0 / 1024.0) << "MiB\n"
                      << "     Free memory " << (mem_free / 1024.0 / 1024.0) << "MiB\n"
                      << "     Total memory " << (mem_total / 1024.0 / 1024.0) << "MiB\n"
                      << "     Allocated by torch " << (DeviceData(device_id).current_allocated_bytes / 1024.0 / 1024.0)
                      << "MB" << std::endl;
        }

        ReportCudaError(cuda_error, "cuda_allocator");
    }
    ReportCudaError(cudaErrorMemoryAllocation, "cuda_allocator");
}

static void* malloc_async(int64_t size, int device_id)
{
    void* ptr;
    auto strm       = cuda::getCurrentCUDAStream();
    auto cuda_error = cudaMallocAsync(&ptr, size, strm);

    handle_cuda_allocation_error(cuda_error, size, device_id);

    TT_CHECK_CUDA_ERROR(cuda_error);
    CHECK_NOTNULL(ptr);

    return ptr;
}

static void free_async(void* ptr)
{
    TT_CHECK_CUDA_ERROR(cudaFreeAsync(ptr, cuda::getCurrentCUDAStream()));
}

static void* malloc_blocking(int64_t size, int device_id)
{
    void* ptr;
    cudaError_t cuda_error = cudaMalloc(&ptr, size);

    handle_cuda_allocation_error(cuda_error, size, device_id);


    return ptr;
}

static void* malloc_blocking_check_free(int64_t size, int device_id)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free < size * 1.1)
    {
        handle_cuda_allocation_error(cudaErrorMemoryAllocation, size, device_id);
    }

    return malloc_blocking(size, device_id);
}

static void free_blocking(void* ptr)
{
    // not quite sure why the cuda device synchronize is needed
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TT_CHECK_CUDA_ERROR(cudaFree(ptr));
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
    else if (algorithm == AllocatorAlgorithm::CUDA_MALLOC_CHECK_FREE)
    {
        ptr  = malloc_blocking_check_free(size, device_id);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC;
    }
    else
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

        if (log_level >= 3 || (log_level >= 2 && size >= log_level_size_th))
        {
            std::cout << "[Allocate] CUDA:" << getDevice() << " " << (size / 1024.0 / 1024.0) << "MiB (" << ptr
                      << ") Curr. Alloc: " << (d.current_allocated_bytes / (1024.0 * 1024.0)) << " MiB";
            std::cout << std::endl;
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
        if (log_level >= 3 || (log_level >= 2 && size >= log_level_size_th))
        {
            std::cout << "Free CUDA Memory with algo= " << (int)algo << " size " << (size / 1024.0 / 1024.0) << "MiB ("
                      << ptr << ")" << "\n";
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
void* cuda_malloc_pinned(int64_t size)
{
    void* ptr              = nullptr;
    cudaError_t cuda_error = cudaMallocHost(&ptr, size);
    //        cudaError_t cuda_error = cudaErrorInvalidValue;
    if (cuda_error != cudaSuccess)
    {
        if (log_level >= 3)
        {
            std::cout << " Pinned memory allocation of " << (size / 1024.0 / 1024.0)
                      << "MiB failed. Falling back to non-pinned memory...\n";
        }
        ptr = nullptr;
    }
    return ptr;
}

void cuda_pinned_free(void* ptr)
{
    TT_CHECK_CUDA_ERROR(cudaFreeHost(ptr));
}


void CUDACachingAllocator::emptyCache()
{
    CHECK(allocator_initialized);
    // this frees unused values for the async allocator
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

}  // namespace cuda

}  // namespace tinytorch