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

static std::recursive_mutex mu;
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

struct PreallocPerDeviceMemoryData
{
    uint8_t* full_ptr       = nullptr;
    int64_t full_size       = 0;
    int64_t full_alloc_info = 0;
    std::vector<std::pair<uint8_t*, int64_t>> free_blocks;
    std::vector<std::pair<uint8_t*, int64_t>> alloc_blocks;
};

static PreallocPerDeviceMemoryData& PreallocDeviceData(int device_id)
{
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, MAX_DEVICES);
    static PreallocPerDeviceMemoryData data[MAX_DEVICES];
    return data[device_id];
}


static AllocatorAlgorithm algorithm = AllocatorAlgorithm::CUDA_MALLOC_ASYNC;
static int log_level                = 1;
const int64_t log_level_size_th     = 10 * 1024 * 1024;


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

static void CleanFreeList(int device_id){

    auto& data = PreallocDeviceData(device_id);

    std::sort(data.free_blocks.begin(), data.free_blocks.end());


    for (int i = 1; i < data.free_blocks.size(); i++)
    {
        if (data.free_blocks[i - 1].first + data.free_blocks[i - 1].second == data.free_blocks[i].first)
        {
            // std::cout << "merge " << (void*)data.free_blocks[i - 1].first << " + " <<
            // (void*)data.free_blocks[i].first
            //     << std::endl;
            // merge from (i-1) -> i
            data.free_blocks[i].first = data.free_blocks[i - 1].first;
            data.free_blocks[i].second += data.free_blocks[i - 1].second;
            data.free_blocks[i - 1].second = 0;
        }
    }

    // remove empty free blocks
    data.free_blocks.erase(
        std::remove_if(data.free_blocks.begin(), data.free_blocks.end(), [](auto pair) { return pair.second == 0; }),
        data.free_blocks.end());

    // std::cout << "prefree " << ptr
    // << " free size: "
    // << data.free_blocks.size() << std::endl;
}

static void* premalloc(int64_t size, int device_id)
{
    size       = iAlignUp(size, 1024);
    auto& data = PreallocDeviceData(device_id);

    void* result_ptr = nullptr;
    for (auto& f : data.free_blocks)
    {
        if (f.second >= size)
        {
            auto remaining = f.second - size;
            if (remaining < 1024 * 8 && remaining > 0)
            {
                // std::cout << "free block small " << remaining << std::endl;
                // use complete block if the free block would be too small to avoid fragmentation
                size = f.second;
            }


            auto return_ptr = f.first;
            data.alloc_blocks.emplace_back(return_ptr, size);
            f.first += size;
            f.second -= size;



            // std::cout << "premalloc " << (void*)return_ptr << " " << (size / (1024.0 * 1024)) << "MiB" << " free
            // size: "
            // << data.free_blocks.size() << std::endl;
            result_ptr = return_ptr;
            break;
        }
    }

    CleanFreeList(device_id);
    return result_ptr;
}

static void prefree(void* ptr, int device_id)
{
    auto& data         = PreallocDeviceData(device_id);
    int alloc_block_id = -1;
    for (int i = 0; i < data.alloc_blocks.size(); i++)
    {
        if (data.alloc_blocks[i].first == ptr)
        {
            alloc_block_id = i;
            break;
        }
    }

    CHECK_GE(alloc_block_id, 0);
    data.free_blocks.emplace_back(data.alloc_blocks[alloc_block_id]);
    data.alloc_blocks.erase(data.alloc_blocks.begin() + alloc_block_id);
    CleanFreeList(device_id);
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
    else if (algorithm == AllocatorAlgorithm::CUDA_MALLOC)
    {
        ptr  = malloc_blocking(size, device_id);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC;
    }
    else if (algorithm == AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
    {
        ptr  = premalloc(size, device_id);
        info = (uint64_t)AllocatorAlgorithm::CUDA_PRE_ALLOCATE;

        // if not enough was preallocated, try using the default cuda malloc
        if (ptr == nullptr)
        {
            // handle_cuda_allocation_error(cudaErrorMemoryAllocation, size, device_id);
            ptr  = malloc_blocking(size, device_id);
            info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC;
            if (ptr)
            {
                std::cout << "use cuda malloc fallback...\n";
            }
        }
    }
    else
    {
        CHECK(false);
    }


    if (ptr && info != (uint64_t)AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
    {
        auto& d = DeviceData(device_id);
        CHECK(d.allocated_blocks.find(ptr) == d.allocated_blocks.end());
        d.allocated_blocks.insert({ptr, size});
        d.current_allocated_bytes += size;
        d.max_allocated_bytes = std::max(d.current_allocated_bytes, d.max_allocated_bytes);
        CHECK(d.allocated_blocks.find(ptr) != d.allocated_blocks.end());
    }

    if (log_level >= 3 || (log_level >= 2 && size >= log_level_size_th))
    {
        std::cout << "[AllocateCUDA] d" << getDevice() << " (" << ptr << ")" << " size: " << size;
        std::cout << std::endl;
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

    int64_t size = 0;
    if (algo != AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
    {
        auto& d = DeviceData(device_id);
        CHECK(d.allocated_blocks.find(ptr) != d.allocated_blocks.end());
        size = d.allocated_blocks.find(ptr)->second;


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
    else if (algo == AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
    {
        prefree(ptr, device_id);
    }
    else
    {
        CHECK(false);
    }


    if (log_level >= 3 || (log_level >= 2 && size >= log_level_size_th))
    {
        std::cout << "[FreeCUDA] d" << device_id << " (" << ptr << ")";
        if (size > 0)
        {
            std::cout << " size: " << size;
        }

        if (algo == AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
        {
            std::cout << " #free: " << PreallocDeviceData(device_id).free_blocks.size();
        }
        std::cout << std::endl;
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
    std::unique_lock l(mu);
    CHECK(allocator_initialized);
    // this frees unused values for the async allocator
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


int64_t pre_allocate_vram(int64_t requested)
{
    std::unique_lock l(mu);
    int64_t allocated = requested;
    CHECK_NE((int)algorithm, (int)AllocatorAlgorithm::CUDA_PRE_ALLOCATE);

    auto device_id = getDevice();
    auto& data     = PreallocDeviceData(device_id);
    CHECK_EQ((void*)data.full_ptr, nullptr);


    auto old_log = log_level;
    log_level    = 0;
    while (true)
    {
        try
        {
            auto [ptr, info] = cuda_cached_malloc(allocated, device_id);


            if (ptr)
            {
                data.full_ptr        = (uint8_t*)ptr;
                data.full_size       = allocated;
                data.full_alloc_info = info;
                break;
            }
        }
        catch (TinyTorchException e)
        {
            allocated -= 500 * (1024 * 1024);
        }
    }

    log_level = old_log;


    data.free_blocks.push_back({data.full_ptr, data.full_size});


    std::cout << "pre_allocate_vram requested " << ((double)requested / (1024 * 1024)) << "MiB allocated "
              << ((double)allocated / (1024 * 1024)) << "MiB" << std::endl;
    return allocated;
}

void free_preallocate_vram()
{
    std::unique_lock l(mu);
    auto device_id = getDevice();
    auto& data     = PreallocDeviceData(device_id);
    CHECK(data.alloc_blocks.empty());

    if (data.full_ptr)
    {
        std::cout << "free_preallocate_vram\n";
        cuda_cached_free(data.full_ptr, data.full_alloc_info, device_id);
        data.full_ptr  = nullptr;
        data.full_size = 0;
        data.free_blocks.clear();
    }
}

int64_t prealloc_free_list_size()
{
    std::unique_lock l(mu);
    auto device_id = getDevice();
    auto& data     = PreallocDeviceData(device_id);
    return data.free_blocks.size();
}

int64_t prealloc_free_memory()
{
    std::unique_lock l(mu);
    auto device_id = getDevice();
    auto& data     = PreallocDeviceData(device_id);
    int64_t sum    = 0;
    for (auto& b : data.free_blocks)
    {
        sum += b.second;
    }
    return sum;
}
}  // namespace cuda
}  // namespace tinytorch