/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

#ifdef TT_HAS_CUDA

using namespace tinytorch;

namespace
{

// Makes a new non-blocking stream the current stream of the device (no implicit sync with the legacy default stream)
// and restores the old stream on scope exit
struct NonBlockingStream
{
    explicit NonBlockingStream(Device device) : device(device)
    {
        cuda::DeviceGuard dg(device);
        original = cuda::getCurrentCUDAStream();
        TT_CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cuda::setCUDAStreamForThisThread(stream);
    }
    ~NonBlockingStream()
    {
        cuda::DeviceGuard dg(device);
        cudaStreamSynchronize(stream);
        cuda::setCUDAStreamForThisThread(original);
        cudaStreamDestroy(stream);
    }

    Device device;
    cudaStream_t original = nullptr;
    cudaStream_t stream   = nullptr;
};

// Launches every kernel used below once: the first launch of a kernel synchronizes (lazy module loading)
void WarmUp(Device device)
{
    auto t = zeros({16}, TensorOptions(device).dtype(kFloat));
    t.fill_(1.0);
    auto c = t.clone();
    c.copy_(t);
    (void)t.sum().item<float>();
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

}  // namespace

// Blocking copies (non_blocking = false) must be ordered with the work on the current stream, also if the stream
// doesn't synchronize with the legacy default stream.
TEST(TinyTorchCudaStreams, BlockingDeviceToHostWaitsForCurrentStream)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    WarmUp(device);
    NonBlockingStream s(device);

    auto t = zeros({1024 * 1024}, TensorOptions(device).dtype(kFloat));
    TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));

    tttest::ReleaseLater(tttest::StallStream(s.stream), 200);
    t.fill_(5.0);
    auto cpu = t.cpu();
    EXPECT_FLOAT_EQ(cpu.min().item<float>(), 5.0f) << "the copy didn't wait for fill_";
    EXPECT_FLOAT_EQ(cpu.max().item<float>(), 5.0f) << "the copy didn't wait for fill_";

    tttest::ReleaseLater(tttest::StallStream(s.stream), 200);
    t.fill_(7.0);
    EXPECT_FLOAT_EQ(t.sum().item<float>(), 7.0f * t.numel()) << "item() didn't wait for fill_";
}

TEST(TinyTorchCudaStreams, BlockingHostToDeviceDoesNotOvertakeCurrentStream)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    cuda::DeviceGuard dg(device);
    WarmUp(device);
    NonBlockingStream s(device);

    int64_t n     = 1024 * 1024;
    auto options  = TensorOptions(device).dtype(kFloat);
    auto t        = full({n}, 1.0f, options);
    auto snapshot = zeros({n}, options);
    auto twos     = full({n}, 2.0f, TensorOptions(kCPU).dtype(kFloat));
    TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));

    // the pending snapshot must still read t = 1
    tttest::ReleaseLater(tttest::StallStream(s.stream), 200);
    snapshot.copy_(t);
    t.copy_(twos);

    TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(s.stream));
    EXPECT_FLOAT_EQ(snapshot.max().item<float>(), 1.0f) << "the host to device copy overtook the pending snapshot";
    EXPECT_FLOAT_EQ(t.min().item<float>(), 2.0f);
}

TEST(TinyTorchCudaStreams, BlockingPeerToPeerWaitsForSourceStream)
{
    int count = 0;
    if (!tttest::has_cuda() || cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    {
        GTEST_SKIP() << "needs at least 2 GPUs";
    }
    Device d0(kCUDA, 0), d1(kCUDA, 1);
    cuda::DeviceGuard dg(d0);
    WarmUp(d0);
    WarmUp(d1);
    NonBlockingStream s0(d0);
    NonBlockingStream s1(d1);

    Tensor t1;
    {
        cuda::DeviceGuard dg1(d1);
        t1 = zeros({1024 * 1024}, TensorOptions(d1).dtype(kFloat));
        TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(s1.stream));
        tttest::ReleaseLater(tttest::StallStream(s1.stream), 200);
        t1.fill_(5.0);
    }

    auto t0 = t1.to(d0);
    EXPECT_FLOAT_EQ(t0.min().item<float>(), 5.0f) << "the peer copy didn't wait for fill_ on the source device";
}

#endif
