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
// A block freed on stream A may only be reused on stream B after A's work up to the free is done. Almost every block is
// reused on the stream that freed it, which needs nothing. So a free records no event: it only stamps the block with
// the number of frees of its stream so far (free_seq). An event recorded on A at any point after the free covers it, so
// the events of a stream ("marks") are recorded lazily, when another stream takes a block. A lazy mark makes the other
// stream wait for all work queued so far, not only up to the free, so a mark is also recorded every kFreesPerMark
// frees, and at the free of a block of at least kEagerMarkBytes (large blocks, e.g. staging buffers, are the ones that
// other streams take, and one event record is cheap next to their use). A mark covers every free with a stamp <= its
// seq. Re-recording a mark only moves it later on the same stream, so a stream gets a few reused events.
//
// Recording a mark needs the stream: destroy streams that allocated or freed with destroyCUDAStream(), which records a
// final mark and removes the stream, so a new stream that gets the same handle is not mistaken for it.
// implemented in tt_cuda.cu: every device for which the stream is the current stream of this thread gets the default
// stream
void resetCUDAStreamOfThisThread(cudaStream_t stream);

static constexpr int kMarksPerStream     = 8;
static constexpr uint64_t kFreesPerMark  = 16;
static constexpr int64_t kEagerMarkBytes = 1024 * 1024;
static uint64_t next_stream_state_id    = 0;

struct StreamState
{
    struct Mark
    {
        cudaEvent_t event = nullptr;
        uint64_t seq      = 0;
    };

    uint64_t id         = 0;
    int device          = 0;
    cudaStream_t stream = nullptr;
    bool destroyed      = false;

    uint64_t free_seq      = 0;
    uint64_t last_mark_seq = 0;
    bool has_mark          = false;
    std::array<Mark, kMarksPerStream> marks;
    int next_mark = 0;

    // (id of another stream, seq of its mark this stream waited on): a later wait for an older mark is redundant
    std::vector<std::pair<uint64_t, uint64_t>> waited;

    StreamState() = default;
    StreamState(const StreamState&) = delete;
    StreamState& operator=(const StreamState&) = delete;
    ~StreamState()
    {
        // runs when the last block stamped by a destroyed stream is taken, possibly with another device current
        try
        {
            DeviceGuard dg(Device(kCUDA, device));
            for (auto& m : marks)
            {
                if (m.event)
                {
                    TT_LOG_FREE_ERROR(cudaEventDestroy(m.event));
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cout << "[TinyTorch] ~StreamState: " << e.what() << std::endl;
        }
    }

    const Mark& RecordMark()
    {
        CHECK(!destroyed);
        DeviceGuard dg(Device(kCUDA, device));
        auto& m = marks[next_mark];
        if (!m.event)
        {
            TT_CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&m.event, cudaEventDisableTiming));
        }
        TT_CHECK_CUDA_ERROR(cudaEventRecord(m.event, stream));
        m.seq         = free_seq;
        last_mark_seq = free_seq;
        has_mark      = true;
        int recorded  = next_mark;
        next_mark     = (next_mark + 1) % kMarksPerStream;
        return marks[recorded];
    }

    // The earliest mark that covers the free with this stamp, nullptr if there is none
    const Mark* FindMarkCovering(uint64_t stamp) const
    {
        const Mark* best = nullptr;
        for (auto& m : marks)
        {
            if (m.event && m.seq >= stamp && (!best || m.seq < best->seq))
            {
                best = &m;
            }
        }
        return best;
    }

    // Makes this stream wait until the free with this stamp on the other stream is done. Returns false if the stamp is
    // no longer needed by anyone: its stream was destroyed and its work is done. Without that, a block that many
    // short-lived streams freed next to (e.g. the tail of the arena) would collect stamps for the whole run.
    bool WaitFor(StreamState& other, uint64_t stamp)
    {
        const Mark* m = other.FindMarkCovering(stamp);
        if (other.destroyed)
        {
            if (!m)
            {
                // the final mark failed after a CUDA error; this throws that error, or the work of the stream is done
                TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                return false;
            }
            cudaError_t query = cudaEventQuery(m->event);
            if (query == cudaSuccess)
            {
                return false;
            }
            if (query == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady)
            {
                // not an error, the application must not see it in a later cudaGetLastError()
                (void)cudaGetLastError();
            }
            // No entry in waited: no stream can free under this id any more, the entry would only grow the list
            TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, m->event));
            return true;
        }
        if (!m)
        {
            m = &other.RecordMark();
        }
        for (auto& w : waited)
        {
            if (w.first == other.id)
            {
                if (w.second < m->seq)
                {
                    TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, m->event));
                    w.second = m->seq;
                }
                return true;
            }
        }
        TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, m->event));
        waited.push_back({other.id, m->seq});
        return true;
    }
};

// The newest free stamp of every stream that freed (a part of) the block. Stream order makes older stamps of the same
// stream redundant, so this holds at most one entry per stream.
using Stamps = std::vector<std::pair<std::shared_ptr<StreamState>, uint64_t>>;

static void MergeStamps(Stamps& dst, const Stamps& src)
{
    for (auto& s : src)
    {
        bool found = false;
        for (auto& d : dst)
        {
            if (d.first == s.first)
            {
                d.second = std::max(d.second, s.second);
                found    = true;
                break;
            }
        }
        if (!found)
        {
            dst.push_back(s);
        }
    }
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
    std::unordered_map<cudaStream_t, std::shared_ptr<StreamState>> streams;
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

static const std::shared_ptr<StreamState>& GetStreamState(int device_id, cudaStream_t strm)
{
    auto& ptr = PreallocDeviceData(device_id).streams[strm];
    if (!ptr)
    {
        ptr         = std::make_shared<StreamState>();
        ptr->id     = next_stream_state_id++;
        ptr->device = device_id;
        ptr->stream = strm;
    }
    return ptr;
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
        for (auto& [other, stamp] : f.stamps)
        {
            if (other.get() == &state || state.WaitFor(*other, stamp))
            {
                kept.push_back({other, stamp});
            }
        }

        uint8_t* return_ptr = it->first;
        if (remaining > 0)
        {
            // the remainder keeps its stamps: they are still valid for any later user
            data.free_blocks.emplace_hint(std::next(it), return_ptr + size,
                                          PreallocFreeBlock{remaining, std::move(kept)});
        }
        data.free_blocks.erase(it);
        data.free_bytes -= size;
        data.alloc_blocks.emplace(return_ptr, size);

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
        auto state = GetStreamState(device_id, getCUDAStream(Device(kCUDA, device_id)));
        state->free_seq++;
        block.stamps.push_back({state, state->free_seq});
        if (!state->has_mark || size >= kEagerMarkBytes || state->free_seq - state->last_mark_seq >= kFreesPerMark)
        {
            state->RecordMark();
        }
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

static void free_blocking(void* ptr)
{
    // not quite sure why the cuda device synchronize is needed
    TT_LOG_FREE_ERROR(cudaDeviceSynchronize());
    TT_LOG_FREE_ERROR(cudaFree(ptr));
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

    std::unique_lock l(mu);
    CHECK(allocator_initialized);

    int64_t size = 0;
    if (algo != AllocatorAlgorithm::CUDA_PRE_ALLOCATE)
    {
        auto& d = DeviceData(device_id);
        CHECK(d.allocated_blocks.find(ptr) != d.allocated_blocks.end());
        size = d.allocated_blocks.find(ptr)->second;
        d.allocated_blocks.erase(ptr);
        if (algo != AllocatorAlgorithm::CUDA_MALLOC)
        {
            d.current_allocated_bytes -= size;
        }
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

    if (algo == AllocatorAlgorithm::CUDA_MALLOC)
    {
        // The synchronization waits for all work of the device, other threads must not wait for it on the lock (unless
        // the caller holds it too, the mutex is recursive). The memory can't be handed out again before cudaFree, so
        // the block is already forgotten; it still counts as allocated until it is freed.
        l.unlock();
        free_blocking(ptr);
        l.lock();
        DeviceData(device_id).current_allocated_bytes -= size;
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
    auto state = GetStreamState(device_id, getCUDAStream(Device(kCUDA, device_id)));
    state->free_seq++;
    data.free_blocks.emplace(data.full_ptr, PreallocFreeBlock{data.full_size, {{state, state->free_seq}}});
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
        TT_LOG_FREE_ERROR(cudaDeviceSynchronize());
        cuda_cached_free(data.full_ptr, data.full_alloc_info, device_id);
        data.full_ptr   = nullptr;
        data.full_size  = 0;
        data.free_bytes = 0;
        data.free_blocks.clear();
    }
}

cudaError_t destroyCUDAStream(cudaStream_t stream)
{
    // Under the lock: an allocation of another thread must not register the handle again between the final mark and
    // the destruction.
    std::unique_lock l(mu);
    // a free of this thread after the destruction must not register the dead handle again
    resetCUDAStreamOfThisThread(stream);
    for (int device_id = 0; device_id < MAX_DEVICES; ++device_id)
    {
        auto& streams = PreallocDeviceData(device_id).streams;
        auto it       = streams.find(stream);
        if (it == streams.end())
        {
            continue;
        }
        auto& state = *it->second;
        try
        {
            // blocks freed on this stream are still stamped with it: a final mark covers all its frees
            if (!state.has_mark || state.last_mark_seq < state.free_seq)
            {
                state.RecordMark();
            }
        }
        catch (const std::exception& e)
        {
            // only fails after a CUDA error, then the device can't be used anyway
            std::cout << "[TinyTorch] destroyCUDAStream: " << e.what() << std::endl;
        }
        state.destroyed = true;
        uint64_t id     = state.id;
        streams.erase(it);
        for (auto& [handle, other] : streams)
        {
            auto& w = other->waited;
            w.erase(std::remove_if(w.begin(), w.end(), [id](const auto& e) { return e.first == id; }), w.end());
        }
    }
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