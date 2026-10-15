/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

#ifdef TT_HAS_CUDA

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

// Blocks the stream until the returned flag is set
std::atomic<bool>* StallStream(cudaStream_t stream)
{
    auto release = new std::atomic<bool>(false);
    TT_CHECK_CUDA_ERROR(cudaLaunchHostFunc(
        stream,
        [](void* p)
        {
            while (!static_cast<std::atomic<bool>*>(p)->load())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        },
        release));
    return release;
}

void ReleaseLater(std::atomic<bool>* release, int ms)
{
    std::thread(
        [release, ms]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            *release = true;
        })
        .detach();
}

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
    new Tensor(input->index_select(0, *index));
    if (cudaDeviceSynchronize() == cudaSuccess)
    {
        std::cout << "failed to trigger a cuda error" << std::endl;
        std::_Exit(2);
    }
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
        ReleaseLater(StallStream(stream_a), 200);
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
