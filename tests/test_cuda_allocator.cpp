/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

#ifdef TT_HAS_CUDA

#    include <algorithm>
#    include <atomic>
#    include <chrono>
#    include <cstdlib>
#    include <iostream>
#    include <optional>
#    include <string>
#    include <thread>

#    ifdef __linux__
#        include <sys/prctl.h>
#    endif

using namespace tinytorch;

namespace
{

// Called in the child of a death test. With a piped core_pattern (e.g. apport) the kernel runs the core dump handler
// even with ulimit -c 0, which adds about a second to every expected crash.
void DisableCoreDumps()
{
#    ifdef __linux__
    prctl(PR_SET_DUMPABLE, 0);
#    endif
}

// Puts the current device into a sticky CUDA error state (illegal memory access, index_select doesn't check the index
// range), like a crashed kernel. Every following CUDA call on this device returns the error. Only use this in the
// child process of a death test.
void TriggerStickyCudaError()
{
    Device device(kCUDA, cuda::getDevice());
    // leaked, only the tensor of the test should be freed after the error
    auto input = new Tensor(zeros({16}, TensorOptions(device).dtype(kFloat)));
    auto index = new Tensor(full({1}, float(int64_t(1) << 40), TensorOptions(device).dtype(kLong)));
    try
    {
        new Tensor(input->index_select(0, *index));
    }
    catch (const std::exception&)
    {
        // debug builds synchronize after the launch (CUDA_SYNC_CHECK_ERROR) and throw already here
    }
    if (cudaDeviceSynchronize() == cudaSuccess)
    {
        std::cout << "failed to trigger a cuda error" << std::endl;
        std::_Exit(2);
    }
}

// emptyCache() is public and may be called before anything was allocated. Runs in a fresh death test child.
void EmptyCacheInFreshProcess()
{
    DisableCoreDumps();
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    cuda::CUDACachingAllocator::emptyCache();
    std::_Exit(0);
}

Tensor* tensor_freed_at_exit = nullptr;

void FreeTensorAtExit()
{
    delete tensor_freed_at_exit;
    tensor_freed_at_exit = nullptr;
}

void AllocateTensorFreedAtExit()
{
    DisableCoreDumps();
    // registered before the first allocation creates the allocator state -> runs after that state was destroyed
    std::atexit(FreeTensorAtExit);
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    for (int i = 0; i < 100; ++i)
    {
        // leaked: many blocks, so that an access to a destroyed block map is noticed
        new Tensor(empty({1024 + i}, TensorOptions(device).dtype(kFloat)));
    }
    tensor_freed_at_exit = new Tensor(empty({1024}, TensorOptions(device).dtype(kFloat)));
    std::exit(0);
}

}  // namespace

namespace
{
// The arena, the allocator mode and the streams of a test of the pre-allocate allocator. The destructor frees them also
// when an ASSERT ends the test early: a leftover arena would fail every following test of the binary.
struct PreallocTest
{
    Device device{kCUDA, 0};
    int64_t n             = 64 * 1024;
    TensorOptions options = TensorOptions(Device(kCUDA, 0)).dtype(kFloat);
    cudaStream_t original = nullptr;
    std::optional<cuda::AllocatorAlgorithmGuard> algorithm_guard;
    bool has_arena = false;
    std::vector<cudaStream_t> streams;
    // hold back the work of stalled streams, released by the test in this order
    std::vector<std::atomic<bool>*> gates;
    // recorded after the work on freed memory that the gates hold back
    std::vector<cudaEvent_t> pending;

    // returns false if the arena could not be allocated
    bool Init()
    {
        original = cuda::getCurrentCUDAStream();
        // load the kernels before stalling a stream: the first launch of a kernel (lazy module loading) synchronizes
        (void)full({n}, 1.0f, options).min().item<float>();
        TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        has_arena = cuda::pre_allocate_vram(64 * 1024 * 1024) >= 64 * 1024 * 1024;
        algorithm_guard.emplace(cuda::AllocatorAlgorithm::CUDA_PRE_ALLOCATE);
        // every allocation of the test must come from the arena
        cuda::prealloc_allow_fallback_cudamalloc(false);
        return has_arena;
    }

    ~PreallocTest()
    {
        Release();
        cuda::setCUDAStreamForThisThread(original);
        (void)cudaDeviceSynchronize();
        for (auto e : pending)
        {
            (void)cudaEventDestroy(e);
        }
        for (auto s : streams)
        {
            (void)cuda::destroyCUDAStream(s);
        }
        cuda::prealloc_allow_fallback_cudamalloc(true);
        algorithm_guard.reset();
        if (has_arena)
        {
            cuda::free_preallocate_vram();
        }
    }

    cudaStream_t NewStream()
    {
        cudaStream_t s;
        TT_CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        streams.push_back(s);
        return s;
    }

    void DestroyStream(cudaStream_t s)
    {
        streams.erase(std::find(streams.begin(), streams.end(), s));
        TT_CHECK_CUDA_ERROR(cuda::destroyCUDAStream(s));
    }

    // Holds back the following work of the stream until Release(). A watchdog releases it after 5 s, so an allocator
    // that blocks the host on the stalled work fails the test instead of hanging it.
    void Stall(cudaStream_t stream)
    {
        auto gate = tttest::StallStream(stream);
        tttest::ReleaseLater(gate, 5000);
        gates.push_back(gate);
    }

    // Releases the gates in the order they were created, with a pause in between, so that the work behind a later
    // gate runs well after the work behind an earlier one
    void Release()
    {
        for (size_t i = 0; i < gates.size(); ++i)
        {
            if (i > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            *gates[i] = true;
        }
        gates.clear();
    }

    // Writes the tensor behind a stall of its stream. A memset, not a kernel: debug builds synchronize after every
    // kernel launch, which would wait for the stall.
    void PendingWrite(cudaStream_t stream, Tensor& x)
    {
        TT_CHECK_CUDA_ERROR(cudaMemsetAsync(x.data_ptr(), 1, x.numel() * x.element_size(), stream));
        cudaEvent_t e;
        TT_CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        TT_CHECK_CUDA_ERROR(cudaEventRecord(e, stream));
        pending.push_back(e);
    }

    // Allocates n floats on the stream, writes them behind a stall of the stream and frees them. Returns the pointer of
    // the freed tensor.
    void* FreeWithPendingWrite(cudaStream_t stream)
    {
        cuda::setCUDAStreamForThisThread(stream);
        Stall(stream);
        auto x    = empty({n}, options);
        void* ptr = x.data_ptr();
        PendingWrite(stream, x);
        return ptr;
    }

    // Allocates numel floats on the stream (expected at the address of a freed tensor) and fills them with 2. Then
    // releases the pending writes to the freed tensor and checks that none of them overwrote y.
    void ExpectReuseWaits(cudaStream_t stream, void* freed_ptr, int64_t numel)
    {
        for (auto e : pending)
        {
            // otherwise the test passes without testing anything
            ASSERT_EQ(cudaEventQuery(e), cudaErrorNotReady) << "test setup: a watchdog released a stall";
        }
        (void)cudaGetLastError();

        cuda::setCUDAStreamForThisThread(stream);
        auto y = empty({numel}, options);
        EXPECT_EQ(y.data_ptr(), freed_ptr) << "test setup: y should reuse the freed memory";
        y.fill_(2.0);
        Release();
        TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "the allocator left an error for the application";
        EXPECT_FLOAT_EQ(y.min().item<float>(), 2.0f) << "a pending write to the freed tensor overwrote y";
        EXPECT_FLOAT_EQ(y.max().item<float>(), 2.0f) << "a pending write to the freed tensor overwrote y";
    }
};
}  // namespace

// An allocation of a block of the pre-allocate allocator on another stream than the one that freed it must wait for the
// work pending on the freeing stream, so that it doesn't overwrite memory that is still used. This must also hold if
// the freeing thread uses many other events in between.
TEST(TinyTorchCudaAllocator, PreallocWaitsForPendingWorkOnFreedMemory)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    PreallocTest t;
    ASSERT_TRUE(t.Init());
    cudaStream_t stream_a = t.NewStream();
    cudaStream_t stream_b = t.NewStream();

    void* x_ptr = t.FreeWithPendingWrite(stream_a);

    // stream b: unrelated work of this thread that uses events
    for (int i = 0; i < 1000; ++i)
    {
        TT_CHECK_CUDA_ERROR(cudaEventRecord(cuda::getNextEvent(), stream_b));
    }
    t.ExpectReuseWaits(stream_b, x_ptr, t.n);
}

// The reuse on another stream waits for the work of the freeing stream up to the free, not for work queued after it.
// Here the later work waits for the host, which only continues after the reuse is done.
TEST(TinyTorchCudaAllocator, PreallocReuseDoesNotWaitForLaterWork)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    PreallocTest t;
    ASSERT_TRUE(t.Init());
    cudaStream_t stream_a = t.NewStream();
    cudaStream_t stream_b = t.NewStream();

    void* x_ptr = nullptr;
    {
        cuda::setCUDAStreamForThisThread(stream_a);
        // not the first free of the stream
        auto w = empty({16}, t.options);
        w      = {};
        auto x = empty({t.n}, t.options);
        x_ptr  = x.data_ptr();
    }
    t.Stall(stream_a);

    cuda::setCUDAStreamForThisThread(stream_b);
    auto y = empty({t.n}, t.options);
    EXPECT_EQ(y.data_ptr(), x_ptr) << "test setup: y should reuse the freed memory";
    y.fill_(2.0);
    auto start = std::chrono::steady_clock::now();
    TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_b));
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(seconds, 2.0) << "the reuse waited for work queued on the freeing stream after the free";
}

// Destroying the freeing stream (with destroyCUDAStream) while its work is pending must not drop the wait, also not for
// a new stream that gets the handle of the destroyed one.
TEST(TinyTorchCudaAllocator, PreallocWaitsForDestroyedStream)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    PreallocTest t;
    ASSERT_TRUE(t.Init());

    cudaStream_t stream_a = t.NewStream();
    void* x_ptr           = t.FreeWithPendingWrite(stream_a);
    t.DestroyStream(stream_a);
    EXPECT_EQ(cuda::getCurrentCUDAStream(), nullptr) << "the destroyed stream is still the current stream";

    cudaStream_t stream_new = t.NewStream();
    if (stream_new != stream_a)
    {
        std::cout << "test setup: the new stream didn't get the handle of the destroyed one, it is only another stream"
                  << std::endl;
    }
    t.ExpectReuseWaits(stream_new, x_ptr, t.n);
}

// The handle of the per-thread default stream means another stream in every thread: a block freed on it by one thread
// and reused on it by another must wait.
TEST(TinyTorchCudaAllocator, PreallocPerThreadDefaultStreamsDiffer)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    PreallocTest t;
    ASSERT_TRUE(t.Init());

    // The other thread stays alive until the end: its per-thread stream must outlive the pending write
    void* x_ptr = nullptr;
    std::atomic<bool> freed{false}, done{false};
    std::thread other(
        [&]()
        {
            cuda::DeviceGuard dg2(Device(kCUDA, 0));
            x_ptr = t.FreeWithPendingWrite(cudaStreamPerThread);
            freed = true;
            while (!done)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    struct JoinGuard
    {
        std::atomic<bool>& done;
        std::thread& thread;
        ~JoinGuard()
        {
            done = true;
            thread.join();
        }
    } join_guard{done, other};
    while (!freed)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    t.ExpectReuseWaits(cudaStreamPerThread, x_ptr, t.n);
}

// Adjacent blocks freed on two different streams are merged (into the block freed first or into the one freed last).
// An allocation of the merged block on a third stream must wait for the pending work of both.
TEST(TinyTorchCudaAllocator, PreallocMergedBlockWaitsForAllStreams)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    for (bool free_first_block_first : {false, true})
    {
        // a lost stamp is only noticed if its work is released last
        for (bool release_first_block_last : {false, true})
        {
            SCOPED_TRACE("free_first_block_first " + std::to_string(free_first_block_first) +
                         " release_first_block_last " + std::to_string(release_first_block_last));
            PreallocTest t;
            ASSERT_TRUE(t.Init());
            cudaStream_t stream_a = t.NewStream();
            cudaStream_t stream_b = t.NewStream();
            cudaStream_t stream_c = t.NewStream();

            t.Stall(release_first_block_last ? stream_b : stream_a);
            t.Stall(release_first_block_last ? stream_a : stream_b);
            void* x1_ptr = nullptr;
            {
                cuda::setCUDAStreamForThisThread(stream_a);
                auto x1 = empty({t.n}, t.options);
                cuda::setCUDAStreamForThisThread(stream_b);
                auto x2 = empty({t.n}, t.options);
                ASSERT_EQ((uint8_t*)x2.data_ptr(), (uint8_t*)x1.data_ptr() + t.n * sizeof(float)) << "test setup";
                x1_ptr = x1.data_ptr();
                t.PendingWrite(stream_a, x1);
                t.PendingWrite(stream_b, x2);
                // x1 is freed on stream a, x2 on stream b
                if (free_first_block_first)
                {
                    cuda::setCUDAStreamForThisThread(stream_a);
                    x1 = {};
                    cuda::setCUDAStreamForThisThread(stream_b);
                }
                else
                {
                    cuda::setCUDAStreamForThisThread(stream_b);
                    x2 = {};
                    cuda::setCUDAStreamForThisThread(stream_a);
                }
            }
            t.ExpectReuseWaits(stream_c, x1_ptr, 2 * t.n);
        }
    }
}

// A stream that already waited for an earlier free of another stream must wait again for a later free: the
// deduplication of waits must not skip it, also after the events of the freeing stream were reused many times.
TEST(TinyTorchCudaAllocator, PreallocWaitsAgainAfterEarlierWait)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    PreallocTest t;
    ASSERT_TRUE(t.Init());
    cudaStream_t stream_a = t.NewStream();
    cudaStream_t stream_b = t.NewStream();

    // stream b waits for a free of stream a. The free is held back, so the wait is not skipped as already done.
    t.Stall(stream_a);
    {
        cuda::setCUDAStreamForThisThread(stream_a);
        auto x = empty({t.n}, t.options);
    }
    {
        cuda::setCUDAStreamForThisThread(stream_b);
        auto y = empty({t.n}, t.options);
    }
    t.Release();
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // many frees on stream a: the pooled events are recorded again and again
    cuda::setCUDAStreamForThisThread(stream_a);
    for (int i = 0; i < 300; ++i)
    {
        auto x = empty({t.n}, t.options);
    }

    void* x_ptr = t.FreeWithPendingWrite(stream_a);
    t.ExpectReuseWaits(stream_b, x_ptr, t.n);
}

// Random allocations and frees on several streams: live blocks never overlap, and after freeing everything the arena is
// one free block again.
TEST(TinyTorchCudaAllocator, PreallocFreeListStaysConsistent)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    cuda::DeviceGuard dg(Device(kCUDA, 0));
    PreallocTest t;
    ASSERT_TRUE(t.Init());
    int64_t arena                     = cuda::prealloc_free_memory();
    std::vector<cudaStream_t> streams = {t.NewStream(), t.NewStream(), t.NewStream()};
    {
        std::vector<Tensor> live;
        uint64_t rng = 12345;
        auto next    = [&]() { return rng = rng * 6364136223846793005ull + 1442695040888963407ull; };
        for (int i = 0; i < 5000; ++i)
        {
            cuda::setCUDAStreamForThisThread(streams[(next() >> 33) % streams.size()]);
            if (live.size() < 64 && (next() >> 33) % 3 != 0)
            {
                int64_t bytes = 1 + (next() >> 33) % (512 * 1024);
                live.push_back(empty({bytes}, TensorOptions(t.device).dtype(kUInt8)));
            }
            else if (!live.empty())
            {
                live.erase(live.begin() + (next() >> 33) % live.size());
            }

            std::vector<std::pair<uint8_t*, uint8_t*>> ranges;
            for (auto& l : live)
            {
                auto p = (uint8_t*)l.data_ptr();
                ranges.push_back({p, p + l.numel()});
            }
            std::sort(ranges.begin(), ranges.end());
            for (size_t j = 1; j < ranges.size(); ++j)
            {
                ASSERT_LE(ranges[j - 1].second, ranges[j].first) << "overlapping blocks at step " << i;
            }
        }
    }
    EXPECT_EQ(cuda::prealloc_free_memory(), arena);
    EXPECT_EQ(cuda::prealloc_free_list_size(), 1);
}

// free_preallocate_vram must wait for the pending work of all streams on the (freed) blocks of the arena before the
// arena is freed. With an arena from cudaMallocAsync, the free is otherwise only ordered on the current stream.
TEST(TinyTorchCudaAllocator, FreePreallocWaitsForPendingWorkOnBlocks)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    int pools_supported = 0;
    TT_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, 0));
    if (!pools_supported)
    {
        GTEST_SKIP() << "no memory pool support";
    }
    cudaStream_t original = cuda::getCurrentCUDAStream();

    int64_t n    = 1024 * 1024;
    auto options = TensorOptions(device).dtype(kFloat);
    empty({n}, options).fill_(1.0);
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    {
        cuda::AllocatorAlgorithmGuard algorithm_guard(cuda::AllocatorAlgorithm::CUDA_MALLOC_ASYNC);
        ASSERT_GE(cuda::pre_allocate_vram(64 * 1024 * 1024), 64 * 1024 * 1024);
    }

    cudaStream_t stream_a;
    TT_CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));
    {
        cuda::AllocatorAlgorithmGuard algorithm_guard(cuda::AllocatorAlgorithm::CUDA_PRE_ALLOCATE);
        cuda::setCUDAStreamForThisThread(stream_a);
        tttest::ReleaseLater(tttest::StallStream(stream_a), 200);
        {
            auto x = empty({n}, options);
            x.fill_(1.0);
        }
        cuda::setCUDAStreamForThisThread(original);
    }

    cuda::free_preallocate_vram();
    EXPECT_EQ(cudaStreamQuery(stream_a), cudaSuccess) << "the arena was freed while work on it was still pending";

    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TT_CHECK_CUDA_ERROR(cuda::destroyCUDAStream(stream_a));
}

// A failed allocation is not a sticky error: the caller can catch the exception and continue (e.g. with a smaller
// batch). The failed cudaMalloc/cudaMallocAsync must not stay the last error, otherwise a later cudaGetLastError() of
// the application reports the out of memory again.
TEST(TinyTorchCudaAllocator, OutOfMemoryDoesNotLeaveLastError)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    (void)cudaGetLastError();

    int old_log = cuda::get_allocator_log_level();
    cuda::set_allocator_log_level(0);
    int64_t huge = int64_t(1) << 50;
    for (auto algo : {cuda::AllocatorAlgorithm::CUDA_MALLOC, cuda::AllocatorAlgorithm::CUDA_MALLOC_ASYNC,
                      cuda::AllocatorAlgorithm::CUDA_MALLOC_CHECK_FREE})
    {
        cuda::AllocatorAlgorithmGuard guard(algo);
        try
        {
            auto t = empty({huge}, TensorOptions(device).dtype(kUInt8));
            ADD_FAILURE() << "allocation of 1 PiB succeeded";
        }
        catch (const TinyTorchException& e)
        {
            EXPECT_EQ(e.status, TinyTorchExceptionStatus::OutOfMemoryGPU) << e.what();
        }
        EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "algorithm " << (int)algo;
    }

    // pre_allocate_vram retries with smaller sizes until an allocation succeeds
    size_t mem_free = 0, mem_total = 0;
    TT_CHECK_CUDA_ERROR(cudaMemGetInfo(&mem_free, &mem_total));
    int64_t allocated = cuda::pre_allocate_vram(int64_t(mem_free) + int64_t(600) * 1024 * 1024);
    EXPECT_GT(allocated, 0);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "pre_allocate_vram";
    cuda::free_preallocate_vram();
    cuda::set_allocator_log_level(old_log);
}

TEST(TinyTorchCudaAllocator, EmptyCacheBeforeFirstAllocation)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    // threadsafe: the child re-executes the binary and only runs this test, so nothing was allocated before
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(EmptyCacheInFreshProcess(), ::testing::ExitedWithCode(0), "");
}

// The allocator state must outlive the static objects of the application: a tensor freed at exit (e.g. held by a
// global) after the statics of the allocator were destroyed must not access destroyed state.
TEST(TinyTorchCudaAllocator, FreeAfterStaticDestruction)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(AllocateTensorFreedAtExit(), ::testing::ExitedWithCode(0), "");
}

// cudaMallocAsync is stream ordered: the arena of pre_allocate_vram may get memory that was freed on the current stream
// while work on it is still pending there. A block of the arena allocated on another stream must wait for that work.
TEST(TinyTorchCudaAllocator, PreallocArenaFromMallocAsyncIsStreamOrdered)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    int pools_supported = 0;
    TT_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, 0));
    if (!pools_supported)
    {
        GTEST_SKIP() << "no memory pool support";
    }
    cudaStream_t original = cuda::getCurrentCUDAStream();

    // the work on the stalled stream is a memset, not a kernel: debug builds synchronize after every kernel launch
    int64_t bytes = 64 * 1024 * 1024;
    auto options  = TensorOptions(device).dtype(kUInt8);
    // load the kernels used below before stalling a stream: the first launch of a kernel synchronizes
    (void)full({16}, 1.0f, TensorOptions(device).dtype(kFloat)).min().item<float>();
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cudaStream_t stream_a, stream_b;
    TT_CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking));
    TT_CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking));

    void* x_ptr = nullptr;
    {
        cuda::AllocatorAlgorithmGuard algorithm_guard(cuda::AllocatorAlgorithm::CUDA_MALLOC_ASYNC);
        cuda::setCUDAStreamForThisThread(stream_a);
        tttest::ReleaseLater(tttest::StallStream(stream_a), 300);
        {
            auto x = empty({bytes}, options);
            x_ptr  = x.data_ptr();
            TT_CHECK_CUDA_ERROR(cudaMemsetAsync(x_ptr, 1, bytes, stream_a));
        }
        // the pool may hand out the memory of x again, x = 1 is still pending on stream a
        ASSERT_GE(cuda::pre_allocate_vram(bytes), bytes);
    }

    {
        cuda::AllocatorAlgorithmGuard algorithm_guard(cuda::AllocatorAlgorithm::CUDA_PRE_ALLOCATE);
        cuda::setCUDAStreamForThisThread(stream_b);
        auto y = empty({bytes}, options);
        if (y.data_ptr() != x_ptr)
        {
            std::cout << "test setup: the arena doesn't reuse the memory of x, the test is not effective" << std::endl;
        }
        TT_CHECK_CUDA_ERROR(cudaMemsetAsync(y.data_ptr(), 2, bytes, stream_b));
        TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_b));
        TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_a));
        EXPECT_EQ((int)y.cpu().min().item<uint8_t>(), 2) << "the pending x = 1 overwrote y";
        y = {};
    }
    cuda::setCUDAStreamForThisThread(original);
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cuda::free_preallocate_vram();
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    TT_CHECK_CUDA_ERROR(cuda::destroyCUDAStream(stream_a));
    TT_CHECK_CUDA_ERROR(cuda::destroyCUDAStream(stream_b));
}

// Memory pool support is queried for the device that is allocated on
TEST(TinyTorchCudaAllocator, MallocAsyncPerDevice)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    int num_devices = 0;
    TT_CHECK_CUDA_ERROR(cudaGetDeviceCount(&num_devices));
    cuda::AllocatorAlgorithmGuard algorithm_guard(cuda::AllocatorAlgorithm::CUDA_MALLOC_ASYNC);
    for (int d = 0; d < num_devices; ++d)
    {
        int pools_supported = 0;
        TT_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, d));
        Device device(kCUDA, d);
        cuda::DeviceGuard dg(device);
        auto t = full({1024}, 3.0f, TensorOptions(device).dtype(kFloat));
        EXPECT_EQ(t.AllocatorInfo(), (uint64_t)(pools_supported ? cuda::AllocatorAlgorithm::CUDA_MALLOC_ASYNC
                                                                : cuda::AllocatorAlgorithm::CUDA_MALLOC))
            << "device " << d;
        EXPECT_FLOAT_EQ(t.max().item<float>(), 3.0f);
    }
}

// Freeing memory after a CUDA error (e.g. a crashed kernel) must not throw: ~StorageImpl is noexcept, so an exception
// calls std::terminate, also while the exception for the original error unwinds the stack.
class TinyTorchCudaFreeAfterError : public ::testing::TestWithParam<int>
{
};

void FreeAfterCudaError(int mode)
{
    DisableCoreDumps();
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    if (mode == 2)
    {
        if (cuda::pre_allocate_vram(64 * 1024 * 1024) < 64 * 1024 * 1024) std::_Exit(3);
    }

    std::optional<cuda::AllocatorAlgorithmGuard> algorithm_guard;
    if (mode == 0) algorithm_guard.emplace(cuda::AllocatorAlgorithm::CUDA_MALLOC);
    if (mode == 1) algorithm_guard.emplace(cuda::AllocatorAlgorithm::CUDA_MALLOC_ASYNC);
    if (mode == 2) algorithm_guard.emplace(cuda::AllocatorAlgorithm::CUDA_PRE_ALLOCATE);

    auto options = TensorOptions(device).dtype(kFloat);
    if (mode == 3) options = TensorOptions(kCPU).dtype(kFloat).pinned_memory(true);
    {
        auto t = empty({1024}, options);
        if (mode == 3 && !t.options().pinned_memory_) std::_Exit(4);
        TriggerStickyCudaError();
    }
    std::_Exit(0);
}

TEST_P(TinyTorchCudaFreeAfterError, DoesNotTerminate)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(FreeAfterCudaError(GetParam()), ::testing::ExitedWithCode(0), "");
}

std::string FreeModeName(const ::testing::TestParamInfo<int>& info)
{
    const char* names[] = {"CudaMalloc", "CudaMallocAsync", "PreAllocate", "Pinned"};
    return names[info.param];
}

INSTANTIATE_TEST_SUITE_P(TinyTorchCudaAllocator, TinyTorchCudaFreeAfterError, ::testing::Values(0, 1, 2, 3),
                         FreeModeName);

#endif
