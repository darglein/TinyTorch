/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

#ifdef TT_HAS_CUDA

#    include <cstdlib>
#    include <iostream>
#    include <optional>
#    include <string>

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

// The pre-allocate allocator stores an event with every freed block. An allocation of that block on another stream
// must wait on it, so that it doesn't overwrite memory that is still used by work pending on the freeing stream. The
// event must stay valid until then, also if the freeing thread uses many other events in between.
TEST(TinyTorchCudaAllocator, PreallocWaitsForPendingWorkOnFreedMemory)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    cudaStream_t original = cuda::getCurrentCUDAStream();

    int64_t n    = 1024 * 1024;
    auto options = TensorOptions(device).dtype(kFloat);

    // load the kernels before stalling a stream: the first launch of a kernel (lazy module loading) synchronizes
    empty({n}, options).fill_(1.0);
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    ASSERT_GE(cuda::pre_allocate_vram(64 * 1024 * 1024), 64 * 1024 * 1024);
    {
        cuda::AllocatorAlgorithmGuard algorithm_guard(cuda::AllocatorAlgorithm::CUDA_PRE_ALLOCATE);

        cudaStream_t stream_a, stream_b;
        TT_CHECK_CUDA_ERROR(cudaStreamCreate(&stream_a));
        TT_CHECK_CUDA_ERROR(cudaStreamCreate(&stream_b));

        // stream a: x = 1 is pending (behind the stall) when x is freed
        cuda::setCUDAStreamForThisThread(stream_a);
        // released long after the host side of this test is done
        tttest::ReleaseLater(tttest::StallStream(stream_a), 200);
        void* x_ptr = nullptr;
        {
            auto x = empty({n}, options);
            x_ptr  = x.data_ptr();
            x.fill_(1.0);
        }

        // stream b: unrelated work of this thread that uses events
        cuda::setCUDAStreamForThisThread(stream_b);
        for (int i = 0; i < 1000; ++i)
        {
            TT_CHECK_CUDA_ERROR(cudaEventRecord(cuda::getNextEvent(), stream_b));
        }

        // stream b: reuses the memory of x. The allocator must make stream b wait until x = 1 is done.
        auto y = empty({n}, options);
        EXPECT_EQ(y.data_ptr(), x_ptr) << "test setup: y should reuse the memory of x";
        y.fill_(2.0);

        TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_b));
        TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_a));
        EXPECT_FLOAT_EQ(y.min().item<float>(), 2.0f) << "the pending x = 1 overwrote y";
        EXPECT_FLOAT_EQ(y.max().item<float>(), 2.0f) << "the pending x = 1 overwrote y";

        y = {};
        cuda::setCUDAStreamForThisThread(original);
        TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        TT_CHECK_CUDA_ERROR(cudaStreamDestroy(stream_a));
        TT_CHECK_CUDA_ERROR(cudaStreamDestroy(stream_b));
    }
    cuda::free_preallocate_vram();
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
    TT_CHECK_CUDA_ERROR(cudaStreamDestroy(stream_a));
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
    TT_CHECK_CUDA_ERROR(cudaStreamDestroy(stream_a));
    TT_CHECK_CUDA_ERROR(cudaStreamDestroy(stream_b));
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
