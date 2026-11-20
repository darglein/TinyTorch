/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

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

// never destroyed, see DeviceData()
static std::recursive_mutex& mu   = *new std::recursive_mutex;
static bool allocator_initialized = false;
// per device: -1 = not queried yet, 0 = no memory pool support, 1 = cudaMallocAsync supported
static int malloc_async_supported[MAX_DEVICES];

struct PerDeviceMemoryData
{
    std::map<void*, int64_t> allocated_blocks;

    int64_t current_allocated_bytes = 0;
    int64_t max_allocated_bytes     = 0;
};

// The allocator state is never destroyed: tensors held by static objects of the application are freed at exit, possibly
// after the static objects of this library were destroyed
static PerDeviceMemoryData& DeviceData(int device_id)
{
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, MAX_DEVICES);
    static PerDeviceMemoryData* data = new PerDeviceMemoryData[MAX_DEVICES];
    return data[device_id];
}

static PerDeviceMemoryData& PinnedMemoryData()
{
    static PerDeviceMemoryData* data = new PerDeviceMemoryData;
    return *data;
}

// Frees run in destructors (also after a CUDA error, e.g. a crashed kernel), so they must not throw
static void LogFreeError(cudaError_t error, const char* function)
{
    if (error != cudaSuccess)
    {
        std::cout << "[TinyTorch] CUDA Error " << cudaGetErrorString(error) << " in function " << function << std::endl;
    }
}
#define TT_LOG_FREE_ERROR(cudaFunction) LogFreeError(cudaFunction, #cudaFunction)

// Stream ordering of the pre-allocated arena.
//
// A block freed on stream A may only be reused on stream B after A's work up to the free is done, so a free records an
// event on its stream and B waits on it. Almost every block is reused on the stream that freed it, which needs no wait.
// The events come from a pool: creating and destroying an event per free cost more than the rest of the allocator,
// recording a reused one is cheap. An event returns to the pool when no free block holds it any more, so it is never
// recorded again while a block still needs it.
//
// Streams are told apart by their handle, the per-thread default stream also by its thread. Destroy streams that
// allocated or freed with destroyCUDAStream(): a new stream that gets the same handle must not be mistaken for the
// destroyed one, which would skip its waits.

// implemented in tt_cuda.cu: every device for which the stream is the current stream of this thread gets the default
// stream
void resetCUDAStreamOfThisThread(cudaStream_t stream);

static uint64_t next_stream_state_id = 0;

// never destroyed, like the rest of the allocator state
static std::vector<cudaEvent_t>& FreeEventPool(int device_id)
{
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, MAX_DEVICES);
    static auto* pools = new std::vector<cudaEvent_t>[MAX_DEVICES];
    return pools[device_id];
}

struct FreeEvent
{
    cudaEvent_t event = nullptr;
    int device        = 0;
};

// Releasing it returns the event to the pool. Only released under the allocator lock.
using FreeEventPtr = std::shared_ptr<FreeEvent>;

static FreeEventPtr AcquireFreeEvent(int device_id)
{
    auto& pool         = FreeEventPool(device_id);
    cudaEvent_t event = nullptr;
    if (pool.empty())
    {
        DeviceGuard dg(Device(kCUDA, device_id));
        TT_CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    else
    {
        event = pool.back();
        pool.pop_back();
    }
    return FreeEventPtr(new FreeEvent{event, device_id},
                        [](FreeEvent* e)
                        {
                            try
                            {
                                FreeEventPool(e->device).push_back(e->event);
                            }
                            catch (const std::exception&)
                            {
                                TT_LOG_FREE_ERROR(cudaEventDestroy(e->event));
                            }
                            delete e;
                        });
}

struct StreamState
{
    uint64_t id         = 0;
    cudaStream_t stream = nullptr;
    uint64_t free_seq   = 0;

    // (id of another stream, seq of its free this stream waited for): a wait for an earlier free is redundant
    std::vector<std::pair<uint64_t, uint64_t>> waited;
};

// A free of (a part of) a block: the number of frees of its stream so far, and the event recorded at the free
struct Stamp
{
    std::shared_ptr<StreamState> stream;
    uint64_t seq;
    FreeEventPtr event;
};

// Stream order makes an older free of the same stream redundant, so this holds at most one stamp per stream
using Stamps = std::vector<Stamp>;

static void MergeStamps(Stamps& dst, const Stamps& src)
{
    for (auto& s : src)
    {
        bool found = false;
        for (auto& d : dst)
        {
            if (d.stream == s.stream)
            {
                if (s.seq > d.seq)
                {
                    d = s;
                }
                found = true;
                break;
            }
        }
        if (!found)
        {
            dst.push_back(s);
        }
    }
}

// Makes the stream wait until the free of the stamp is done. Returns false if the stamp is no longer needed by anyone
// because that work is done: a block that many streams freed next to (e.g. the tail of the arena) would otherwise
// collect their stamps.
static bool WaitFor(StreamState& state, const Stamp& stamp)
{
    for (auto& w : state.waited)
    {
        if (w.first == stamp.stream->id && w.second >= stamp.seq)
        {
            return true;
        }
    }
    cudaError_t query = cudaEventQuery(stamp.event->event);
    if (query == cudaSuccess)
    {
        return false;
    }
    if (query == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady)
    {
        // not an error, the application must not see it in a later cudaGetLastError()
        (void)cudaGetLastError();
    }
    TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(state.stream, stamp.event->event));
    for (auto& w : state.waited)
    {
        if (w.first == stamp.stream->id)
        {
            w.second = stamp.seq;
            return true;
        }
    }
    state.waited.push_back({stamp.stream->id, stamp.seq});
    return true;
}

struct PreallocFreeBlock
{
    int64_t size;
    Stamps stamps;
};

struct PreallocPerDeviceMemoryData
{
    uint8_t* full_ptr       = nullptr;
    int64_t full_size       = 0;
    int64_t full_alloc_info = 0;
    int64_t reserved_bytes  = 0;
    // ordered by address: first fit, and merging with the neighbors on free
    std::map<uint8_t*, PreallocFreeBlock> free_blocks;
    int64_t free_bytes = 0;
    std::unordered_map<void*, int64_t> alloc_blocks;
    // the thread is only set for the per-thread default stream, whose handle means another stream in every thread
    std::map<std::pair<cudaStream_t, std::thread::id>, std::shared_ptr<StreamState>> streams;
};

static PreallocPerDeviceMemoryData& PreallocDeviceData(int device_id)
{
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, MAX_DEVICES);
    static PreallocPerDeviceMemoryData* data = new PreallocPerDeviceMemoryData[MAX_DEVICES];
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
    std::unique_lock l(mu);
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
    std::unique_lock l(mu);
    return DeviceData(device_id).current_allocated_bytes;
}

int64_t max_allocated_size(int device_id)
{
    std::unique_lock l(mu);
    return DeviceData(device_id).max_allocated_bytes;
}

void reset_max_allocated_size(int device_id)
{
    std::unique_lock l(mu);
    auto& d               = DeviceData(device_id);
    d.max_allocated_bytes = d.current_allocated_bytes;
}

void add_external_allocated_size(int device_id, int64_t size)
{
    std::unique_lock l(mu);
    auto& d = DeviceData(device_id);
    d.current_allocated_bytes += size;
    CHECK_GE(d.current_allocated_bytes, 0);
    d.max_allocated_bytes = std::max(d.current_allocated_bytes, d.max_allocated_bytes);
}

void set_max_pinned_memory(int64_t size)
{
    std::unique_lock l(mu);
    max_pinned_memory = size;
}

int64_t current_pinned_allocated_size()
{
    std::unique_lock l(mu);
    return PinnedMemoryData().current_allocated_bytes;
}

static void initialize_allocator()
{
    CHECK(!allocator_initialized);
    allocator_initialized = true;
    std::fill(std::begin(malloc_async_supported), std::end(malloc_async_supported), -1);
}

static bool has_malloc_async(int device_id)
{
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, MAX_DEVICES);
    if (malloc_async_supported[device_id] < 0)
    {
        int value = 0;
        TT_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, device_id));
        malloc_async_supported[device_id] = value != 0;
    }
    return malloc_async_supported[device_id] == 1;
}

static void handle_cuda_allocation_error(cudaError_t cuda_error, int64_t size, int device_id)
{
    if (cuda_error == cudaErrorMemoryAllocation)
    {
        if (log_level >= 1)
        {
            size_t mem_free, mem_total;
            auto info_error = cudaMemGetInfo(&mem_free, &mem_total);
            std::cout << " CUDA out of memory!\n"
                      << "     Device              " << device_id << "\n"
                      << "     Allocator           " << (int)algorithm << " \n"
                      << "     info_error          " << info_error << " \n"
                      << "     Tried to allocate   " << size << " B = " << (size / 1024.0 / 1024.0) << " MiB\n"
                      << "     Free memory         " << (mem_free / 1024.0 / 1024.0) << " MiB\n"
                      << "     Total memory        " << (mem_total / 1024.0 / 1024.0) << " MiB\n"
                      << "     Allocated by torch  "
                      << (DeviceData(device_id).current_allocated_bytes / 1024.0 / 1024.0) << " MiB";

            if (algorithm == AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
            {
                std::cout << "\n     Prealloc Freemem     " << (prealloc_free_memory() / 1024.0 / 1024.0) << " MiB\n";
            }

            std::cout << std::endl;
        }

        ReportCudaError(cuda_error, "cuda_allocator");
    }
}

// An out of memory error is not sticky, the caller may catch the exception and continue. It must not stay the last
// error, otherwise a later cudaGetLastError() (e.g. a launch check of the application) reports it again.
static void ClearOutOfMemoryError(cudaError_t cuda_error)
{
    if (cuda_error == cudaErrorMemoryAllocation)
    {
        (void)cudaGetLastError();
    }
}

static void* malloc_async(int64_t size, int device_id)
{
    void* ptr;
    auto strm       = cuda::getCurrentCUDAStream();
    auto cuda_error = cudaMallocAsync(&ptr, size, strm);
    ClearOutOfMemoryError(cuda_error);

    handle_cuda_allocation_error(cuda_error, size, device_id);

    TT_CHECK_CUDA_ERROR(cuda_error);
    CHECK_NOTNULL(ptr);

    return ptr;
}

// Forgets the streams whose (handle, thread) matches: the blocks they freed keep their stamps and events, but a new
// stream that gets the same identity is not mistaken for them. Call under the lock.
template <typename Match>
static void ForgetStreams(Match match)
{
    for (int device_id = 0; device_id < MAX_DEVICES; ++device_id)
    {
        auto& streams = PreallocDeviceData(device_id).streams;
        std::vector<uint64_t> ids;
        for (auto it = streams.begin(); it != streams.end();)
        {
            if (match(it->first))
            {
                ids.push_back(it->second->id);
                it = streams.erase(it);
            }
            else
            {
                ++it;
            }
        }
        if (ids.empty())
        {
            continue;
        }
        for (auto& [key, other] : streams)
        {
            auto& w = other->waited;
            w.erase(std::remove_if(w.begin(), w.end(),
                                   [&](const auto& e) { return std::find(ids.begin(), ids.end(), e.first) != ids.end(); }),
                    w.end());
        }
    }
}

// A new thread may get the id of a thread that exited, so the per-thread default streams of a thread are forgotten when
// it exits
struct ForgetPerThreadStreamsAtThreadExit
{
    ~ForgetPerThreadStreamsAtThreadExit()
    {
        std::unique_lock l(mu);
        auto thread = std::this_thread::get_id();
        ForgetStreams([thread](const auto& key) { return key.second == thread; });
    }
};

static bool IsPerThreadStream(cudaStream_t strm)
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (strm == nullptr)
    {
        return true;
    }
#endif
    return strm == cudaStreamPerThread;
}

static const std::shared_ptr<StreamState>& GetStreamState(int device_id, cudaStream_t strm)
{
    std::thread::id thread;
    if (IsPerThreadStream(strm))
    {
        static thread_local ForgetPerThreadStreamsAtThreadExit forget_at_exit;
        (void)forget_at_exit;
        thread = std::this_thread::get_id();
    }
    auto& ptr              = PreallocDeviceData(device_id).streams[{strm, thread}];
    if (!ptr)
    {
        ptr         = std::make_shared<StreamState>();
        ptr->id     = next_stream_state_id++;
        ptr->stream = strm;
    }
    return ptr;
}

// Records a free on the current stream of the device
static Stamp RecordFree(int device_id)
{
    cudaStream_t strm = getCUDAStream(Device(kCUDA, device_id));
    auto& state       = GetStreamState(device_id, strm);
    auto event        = AcquireFreeEvent(device_id);
    TT_CHECK_CUDA_ERROR(cudaEventRecord(event->event, strm));
    state->free_seq++;
    return {state, state->free_seq, event};
}

static void* premalloc(int64_t initial_size, int device_id)
{
    int64_t size = iAlignUp(initial_size, prealloc_alignment);
    auto& data   = PreallocDeviceData(device_id);

    if (data.free_bytes - size < data.reserved_bytes)
    {
        return nullptr;
    }

    for (auto it = data.free_blocks.begin(); it != data.free_blocks.end(); ++it)
    {
        auto& f = it->second;
        if (f.size < size)
        {
            continue;
        }
        auto remaining = f.size - size;
        if (remaining < 1024 * 8 && remaining > 0)
        {
            // use complete block if the free block would be too small to avoid fragmentation
            size      = f.size;
            remaining = 0;
        }

        auto& state = *GetStreamState(device_id, getCUDAStream(Device(kCUDA, device_id)));
        Stamps kept;
        for (auto& stamp : f.stamps)
        {
            if (stamp.stream.get() == &state || WaitFor(state, stamp))
            {
                kept.push_back(stamp);
            }
        }

        // nothing of the free list changes before every step that can throw is done
        uint8_t* return_ptr = it->first;
        data.alloc_blocks.emplace(return_ptr, size);
        if (remaining > 0)
        {
            try
            {
                // the remainder keeps its stamps: they are still valid for any later user
                data.free_blocks.emplace_hint(std::next(it), return_ptr + size,
                                              PreallocFreeBlock{remaining, std::move(kept)});
            }
            catch (...)
            {
                data.alloc_blocks.erase(return_ptr);
                throw;
            }
        }
        data.free_blocks.erase(it);
        data.free_bytes -= size;

        // make sure the pointer is really aligned
        CHECK_EQ(((uintptr_t)return_ptr) % prealloc_alignment, 0);
        // make sure remaining size is also aligned
        CHECK_EQ(remaining % prealloc_alignment, 0);
        return return_ptr;
    }
    return nullptr;
}

static void prefree(void* ptr, int device_id)
{
    auto& data = PreallocDeviceData(device_id);
    auto a     = data.alloc_blocks.find(ptr);
    CHECK(a != data.alloc_blocks.end());
    int64_t size = a->second;
    data.alloc_blocks.erase(a);

    PreallocFreeBlock block{size, {}};
    try
    {
        block.stamps.push_back(RecordFree(device_id));
    }
    catch (const std::exception& e)
    {
        // only fails after a CUDA error, then the device can't be used anyway
        std::cout << "[TinyTorch] free: " << e.what() << std::endl;
    }

    auto it = data.free_blocks.emplace((uint8_t*)ptr, std::move(block)).first;
    data.free_bytes += size;

    auto next = std::next(it);
    if (next != data.free_blocks.end() && it->first + it->second.size == next->first)
    {
        it->second.size += next->second.size;
        MergeStamps(it->second.stamps, next->second.stamps);
        data.free_blocks.erase(next);
    }
    if (it != data.free_blocks.begin())
    {
        auto prev = std::prev(it);
        if (prev->first + prev->second.size == it->first)
        {
            prev->second.size += it->second.size;
            MergeStamps(prev->second.stamps, it->second.stamps);
            data.free_blocks.erase(it);
        }
    }
}

static void free_async(void* ptr)
{
    TT_LOG_FREE_ERROR(cudaFreeAsync(ptr, cuda::getCurrentCUDAStream()));
}

static void* malloc_blocking(int64_t size, int device_id)
{
    void* ptr              = nullptr;
    cudaError_t cuda_error = cudaMalloc(&ptr, size);
    ClearOutOfMemoryError(cuda_error);

    handle_cuda_allocation_error(cuda_error, size, device_id);
    TT_CHECK_CUDA_ERROR(cuda_error);

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

    if (algorithm == AllocatorAlgorithm::CUDA_MALLOC_ASYNC && has_malloc_async(device_id))
    {
        ptr  = malloc_async(size, device_id);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC_ASYNC;
    }
    else if (algorithm == AllocatorAlgorithm::CUDA_MALLOC_CHECK_FREE)
    {
        ptr  = malloc_blocking_check_free(size, device_id);
        info = (uint64_t)AllocatorAlgorithm::CUDA_MALLOC;
    }
    else if (algorithm == AllocatorAlgorithm::CUDA_MALLOC || algorithm == AllocatorAlgorithm::CUDA_MALLOC_ASYNC)
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
                      << " total " << current_mib << "MiB" << std::endl;
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

    if (algo == AllocatorAlgorithm::CUDA_MALLOC)
    {
        // cudaFree is not documented to wait for work on other streams that may still use the memory. The
        // synchronization waits for all work of the device: other threads must not wait for it on the lock (unless the
        // caller holds it, the mutex is recursive). cudaFree and the bookkeeping stay under the lock, so the peak only
        // counts allocations that coexist.
        TT_LOG_FREE_ERROR(cudaDeviceSynchronize());
    }

    std::unique_lock l(mu);
    CHECK(allocator_initialized);

    int64_t size = 0;
    if (algo != AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
    {
        auto& d = DeviceData(device_id);
        CHECK(d.allocated_blocks.find(ptr) != d.allocated_blocks.end());
        size = d.allocated_blocks.find(ptr)->second;
        d.allocated_blocks.erase(ptr);
        d.current_allocated_bytes -= size;
    }



    if (algo == AllocatorAlgorithm::CUDA_MALLOC_ASYNC)
    {
        free_async(ptr);
    }
    else if (algo == AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
    {
        prefree(ptr, device_id);
    }
    else
    {
        CHECK(algo == AllocatorAlgorithm::CUDA_MALLOC);
        TT_LOG_FREE_ERROR(cudaFree(ptr));
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

    void* ptr = nullptr;
    // cudaError_t cuda_error = cudaMallocHost(&ptr, size);

    ptr                    = malloc(size);
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
        // the caller falls back to non-pinned memory, so clear the (non-sticky) error
        (void)cudaGetLastError();
        free(ptr);
        ptr = nullptr;
    }
    return ptr;
}

void cuda_pinned_free(void* ptr, int64_t size)
{
    std::unique_lock l(mu);
    if (ptr)
    {
        // TT_CHECK_CUDA_ERROR(cudaFreeHost(ptr));
        TT_LOG_FREE_ERROR(cudaHostUnregister(ptr));
        free(ptr);

        auto& d = PinnedMemoryData();
        d.current_allocated_bytes -= size;
    }
}


void CUDACachingAllocator::emptyCache()
{
    std::unique_lock l(mu);
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
        catch (const TinyTorchException&)
        {
        }

        allocated -= 500 * (1024 * 1024);

        if (allocated <= 0)
        {
            log_level = old_log;
            return 0;
        }
    }

    log_level = old_log;


    // an arena from cudaMallocAsync is stream ordered: it may reuse memory that pending work on the current stream
    // still uses, so the arena is stamped like a block freed on the current stream
    data.free_blocks.emplace(data.full_ptr, PreallocFreeBlock{data.full_size, {RecordFree(device_id)}});
    data.free_bytes = data.full_size;

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
        // work on any stream may still use (already freed) blocks of the arena
        for (auto& [ptr, block] : data.free_blocks)
        {
            for (auto& stamp : block.stamps)
            {
                TT_LOG_FREE_ERROR(cudaEventSynchronize(stamp.event->event));
            }
        }
        cuda_cached_free(data.full_ptr, data.full_alloc_info, device_id);
        data.full_ptr   = nullptr;
        data.full_size  = 0;
        data.free_bytes = 0;
        data.free_blocks.clear();
    }
}

cudaError_t destroyCUDAStream(cudaStream_t stream)
{
    std::unique_lock l(mu);
    // a free of this thread after the destruction must not register the dead handle again
    resetCUDAStreamOfThisThread(stream);
    ForgetStreams([stream](const auto& key) { return key.first == stream; });
    return cudaStreamDestroy(stream);
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
    return PreallocDeviceData(device_id).free_bytes;
}
void prealloc_print_debug_line()
{
    auto device_id = getDevice();
    auto& data     = PreallocDeviceData(device_id);

    std::cout << "pre_allocate: device " << getDevice() << " free_list_size " << prealloc_free_list_size()
              << " free_mem " << (prealloc_free_memory() - data.reserved_bytes) << " free_mem_mib "
              << ((prealloc_free_memory() - data.reserved_bytes) / 1024.0 / 1024) << " reserved_mib "
              << (data.reserved_bytes / 1024.0 / 1024) << std::endl;
}
void prealloc_allow_fallback_cudamalloc(bool value)
{
    std::unique_lock l(mu);
    prealloc_use_fallback = value;
}
void pre_allocate_set_reserved_memory(int64_t reserve)
{
    std::unique_lock l(mu);
    PreallocDeviceData(getDevice()).reserved_bytes = reserve;
}
int64_t pre_allocate_get_reserved_memory()
{
    std::unique_lock l(mu);
    return PreallocDeviceData(getDevice()).reserved_bytes;
}
}  // namespace cuda
}  // namespace tinytorch