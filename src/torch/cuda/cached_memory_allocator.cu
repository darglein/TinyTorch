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
static constexpr int MAX_DEVICES            = 8;
static constexpr int64_t prealloc_alignment = 1024;
static bool prealloc_use_fallback           = true;

// all pinned mallocs after this will return nullptr
static int64_t max_pinned_memory = 0;

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

static PerDeviceMemoryData& PinnedMemoryData()
{
    static PerDeviceMemoryData data;
    return data;
}

struct PreallocBlock
{
    uint8_t* ptr;
    int64_t size;
    // cudaStream_t last_stream = 0;

    // recorded at the moment of free()
    // -> next usage should wait on this event
    std::vector<std::pair<cudaEvent_t, cudaStream_t>> free_events;
};

bool operator<(const PreallocBlock& lhs, const PreallocBlock& rhs)
{
    return std::make_pair(lhs.ptr, lhs.size) < std::make_pair(rhs.ptr, rhs.size);
}

struct PreallocPerDeviceMemoryData
{
    uint8_t* full_ptr       = nullptr;
    int64_t full_size       = 0;
    int64_t full_alloc_info = 0;
    std::vector<PreallocBlock> free_blocks;
    std::vector<PreallocBlock> alloc_blocks;
};

static PreallocPerDeviceMemoryData& PreallocDeviceData(int device_id)
{
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, MAX_DEVICES);
    static PreallocPerDeviceMemoryData data[MAX_DEVICES];
    return data[device_id];
}


static AllocatorAlgorithm algorithm = AllocatorAlgorithm::CUDA_MALLOC;
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
void set_max_pinned_memory(int64_t size)
{
    max_pinned_memory = size;
}

int64_t current_pinned_allocated_size()
{
    return PinnedMemoryData().current_allocated_bytes;
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

static void CleanFreeList(int device_id)
{
    auto& data = PreallocDeviceData(device_id);

    std::sort(data.free_blocks.begin(), data.free_blocks.end());


    for (int i = 1; i < data.free_blocks.size(); i++)
    {
        auto& b0 = data.free_blocks[i - 1];
        auto& b1 = data.free_blocks[i];

        // merge b0 into b1 ( b0 will be empty)
        if (b0.size > 0 && b0.ptr + b0.size == b1.ptr)
        {
            // std::cout << "merge " << (void*)data.free_blocks[i - 1].first << " + " <<
            // (void*)data.free_blocks[i].first
            //     << std::endl;
            // merge from (i-1) -> i
            b1.ptr = b0.ptr;
            b1.size += b0.size;
            b0.size = 0;

            b1.free_events.insert(b1.free_events.end(), b0.free_events.begin(), b0.free_events.end());
            b0.free_events.clear();

            // if (b0.last_stream != b1.last_stream)
            {
                // auto finished_event = getNextEvent();
                // cudaEventRecord(finished_event, b0.last_stream);
                // cudaStreamWaitEvent(b1.last_stream, finished_event);
            }
        }
    }

    // remove empty free blocks
    data.free_blocks.erase(
        std::remove_if(data.free_blocks.begin(), data.free_blocks.end(), [](auto pair) { return pair.size == 0; }),
        data.free_blocks.end());

    // std::cout << "prefree " << ptr
    // << " free size: "
    // << data.free_blocks.size() << std::endl;
}

static void* premalloc(int64_t initial_size, int device_id)
{
    int64_t size = iAlignUp(initial_size, prealloc_alignment);
    auto& data   = PreallocDeviceData(device_id);

    auto strm        = getCurrentCUDAStream();
    void* result_ptr = nullptr;

    // std::cout << "premalloc " << initial_size << " free " <<  prealloc_free_memory()  << std::endl;
    for (auto& f : data.free_blocks)
    {
        // std::cout << "free " << f.size << std::endl;
        if (f.size >= size)
        {
            auto remaining = f.size - size;
            if (remaining < 1024 * 8 && remaining > 0)
            {
                // std::cout << "free block small " << remaining << std::endl;
                // use complete block if the free block would be too small to avoid fragmentation
                size = f.size;
            }

            for (auto& e : f.free_events)
            {
                if (e.second != strm)
                {
                    // only wait if stream different
                    TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(strm, e.first));
                }
            }

            if (f.free_events.size() > 1)
            {
                // replace the large event list of the free block by just one event on the current stream,
                // beecause we have just waited on all events
                f.free_events.clear();
                auto finished_event = getNextEvent();
                TT_CHECK_CUDA_ERROR(cudaEventRecord(finished_event, strm));
                f.free_events.push_back({finished_event, strm});
            }


            auto return_ptr = f.ptr;

            PreallocBlock out_block;
            out_block.ptr  = return_ptr;
            out_block.size = size;
            out_block.free_events.clear();

            data.alloc_blocks.emplace_back(out_block);
            f.ptr += size;
            f.size -= size;

            // make sure the pointer is really aligned
            CHECK_EQ(((uintptr_t)return_ptr) % prealloc_alignment, 0);

            // make sure remaining size is also aligned
            CHECK_EQ(f.size % prealloc_alignment, 0);



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
        if (data.alloc_blocks[i].ptr == ptr)
        {
            alloc_block_id = i;
            break;
        }
    }


    // std::cout << "prefree " << data.alloc_blocks[alloc_block_id].size << " free " <<  prealloc_free_memory()  <<
    // std::endl;

    {
        DeviceGuard dg(Device(kCUDA, device_id));
        auto finished_event = getNextEvent();
        auto strm           = getCurrentCUDAStream();
        TT_CHECK_CUDA_ERROR(cudaEventRecord(finished_event, strm));
        data.alloc_blocks[alloc_block_id].free_events.push_back({finished_event, strm});
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
    if (size <= 0)
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
        if (prealloc_use_fallback && ptr == nullptr)
        {
            ptr  = malloc_blocking(size, device_id);
            info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC;
            if (ptr)
            {
                std::cout << "use cuda malloc fallback...\n";
            }
        }
        else if (ptr == nullptr)
        {
            handle_cuda_allocation_error(cudaErrorMemoryAllocation, size, device_id);
        }
    }
    else
    {
        CHECK(false) << "invalid allocator: " << (int)algorithm;
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
        double size_mib    = (size / 1024.0 / 1024.0);
        double current_mib = current_allocated_size(getDevice()) / 1024.0 / 1024.0;
        // if (size_mib > 10)
        {
            std::cout << "[AllocateCUDA] d" << getDevice() << " (" << ptr << ")" << " size: " << size_mib << "MiB"
                      << " total " << current_mib << "MiB"<< std::endl;
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
    std::unique_lock l(mu);

    if (max_pinned_memory > 0 && PinnedMemoryData().current_allocated_bytes + size > max_pinned_memory)
    {
        std::cout << "pinned memory limit reached!" << std::endl;
        return nullptr;
    }

    void* ptr              = nullptr;
    // cudaError_t cuda_error = cudaMallocHost(&ptr, size);

    ptr = malloc(size);
    cudaError_t cuda_error = cudaHostRegister(ptr, size, cudaHostRegisterDefault);


    if (ptr && cuda_error == cudaSuccess)
    {
        auto& d = PinnedMemoryData();
        d.current_allocated_bytes += size;
        d.max_allocated_bytes = std::max(d.current_allocated_bytes, d.max_allocated_bytes);
    }


    //    if (size > 1024 * 1024)
    //    {
    //         cudaMallocHost(&ptr, size);
    //        std::cout << "pinned alloc " << (size / 1024.0 / 1024.0) << " MiB" << std::endl;
    //    }
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

void cuda_pinned_free(void* ptr, int64_t size)
{
    if (ptr)
    {
    // TT_CHECK_CUDA_ERROR(cudaFreeHost(ptr));
        TT_CHECK_CUDA_ERROR(cudaHostUnregister(ptr));
        free(ptr);

        auto& d = PinnedMemoryData();
        d.current_allocated_bytes -= size;
    }
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
            allocated        = iAlignUp(allocated, prealloc_alignment);
            auto [ptr, info] = cuda_cached_malloc(allocated, device_id);

            if (ptr)
            {
                CHECK_EQ(((uintptr_t)ptr) % prealloc_alignment, 0);
                data.full_ptr        = (uint8_t*)ptr;
                data.full_size       = allocated;
                data.full_alloc_info = info;
                break;
            }
        }
        catch (TinyTorchException e)
        {
        }

        allocated -= 500 * (1024 * 1024);

        if (allocated <= 0)
        {
            return 0;
        }
    }

    log_level = old_log;


    data.free_blocks.push_back({data.full_ptr, data.full_size});

    if (log_level > 3)
    {
        std::cout << "pre_allocate_vram requested " << ((double)requested / (1024 * 1024)) << "MiB allocated "
                  << ((double)allocated / (1024 * 1024)) << "MiB" << std::endl;
    }
    prealloc_print_debug_line();
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
        if (log_level > 3)
        {
            std::cout << "free_preallocate_vram\n";
        }
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
        sum += b.size;
    }
    return sum;
}
void prealloc_print_debug_line()
{
    std::cout << "pre_allocate: device " << getDevice() << " free_list_size " << prealloc_free_list_size()
              << " free_mem_mib " << (prealloc_free_memory() / 1024.0 / 1024) << std::endl;
}
void prealloc_allow_fallback_cudamalloc(bool value)
{
    prealloc_use_fallback = value;
}
}  // namespace cuda
}  // namespace tinytorch