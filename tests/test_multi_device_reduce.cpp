/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

#ifdef TT_HAS_CUDA
#    include "torch/cuda/multi_device.h"

namespace
{

std::vector<Device> all_cuda_devices()
{
    std::vector<Device> d;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        return d;
    }
    for (int i = 0; i < count; ++i)
    {
        d.push_back(Device(kCUDA, i));
    }
    return d;
}

// Device counts to exercise the reduce: 2..min(available, 4) (the kernel supports up to 4).
std::vector<int> reduce_device_counts()
{
    std::vector<int> counts;
    int avail = 0;
    cudaGetDeviceCount(&avail);
    int maxc = std::min(avail, 4);
    for (int n = 2; n <= maxc; ++n)
    {
        counts.push_back(n);
    }
    // Always return at least one count so the parameterized suite is instantiated (gtest
    // errors on an empty one); the single case GTEST_SKIPs in SetUp when devices are missing.
    if (counts.empty())
    {
        counts.push_back(2);
    }
    return counts;
}

}  // namespace

// Regression tests for the UVA reduce kernels (MultiDeviceTensor::ReduceSumToMainUVA /
// ReduceGradientSumToMainUVA). The ReduceToDevice0 kernel walks a *linear* tid over the
// whole buffer, but a historical bug in MultiGPUInputSimple bounded that index to size(0)
// instead of numel(), so a rank>=2 tensor had only its first size(0) linear elements
// reduced and the rest left untouched. A 1-D tensor (size(0)==numel) or a single-device
// run (kernel not launched) both mask the bug, so these tests use a 2-D shape {4,3}
// (numel=12 != size(0)=4) across 2..4 devices, giving every device a distinct value so
// that every element of the reduced result must equal the per-element sum.
class TinyTorchMultiDeviceReduceParam : public ::testing::TestWithParam<int>
{
   protected:
    static constexpr int64_t kRows = 4;  // size(0)
    static constexpr int64_t kCols = 3;  // numel = kRows * kCols = 12 != size(0)
    std::vector<Device> devices_;

    int num_devices() const { return GetParam(); }

    void SetUp() override
    {
        std::vector<Device> all = all_cuda_devices();
        if (num_devices() > (int)all.size())
        {
            GTEST_SKIP() << "need " << num_devices() << " CUDA devices, have " << all.size();
        }
        devices_ = std::vector<Device>(all.begin(), all.begin() + num_devices());
        if (!tinytorch::cuda::EnableCudaPeerToPeer(devices_))
        {
            GTEST_SKIP() << "P2P not available for " << num_devices() << " devices";
        }
    }
    void TearDown() override
    {
        if (devices_.size() >= 2)
        {
            tinytorch::cuda::DisableCudaPeerToPeer(devices_);
        }
    }

    static Tensor filled(float v, Device d) { return tinytorch::full({kRows, kCols}, v, TensorOptions().device(d)); }
};

INSTANTIATE_TEST_SUITE_P(DeviceCounts, TinyTorchMultiDeviceReduceParam, ::testing::ValuesIn(reduce_device_counts()),
                         [](const ::testing::TestParamInfo<int>& info)
                         { return "devices" + std::to_string(info.param); });

// Enabling twice and disabling twice must not throw or leave a pending error, and disabling must disable every pair
// (with 3+ devices, the pairs without device 0 stayed enabled).
TEST_P(TinyTorchMultiDeviceReduceParam, EnableDisablePeerToPeer)
{
    // enabled in SetUp
    EXPECT_TRUE(tinytorch::cuda::EnableCudaPeerToPeer(devices_));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    tinytorch::cuda::DisableCudaPeerToPeer(devices_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_FALSE(tinytorch::cuda::HasP2PCopy());
    for (size_t i = 0; i < devices_.size(); ++i)
    {
        for (size_t j = 0; j < devices_.size(); ++j)
        {
            if (i == j) continue;
            tinytorch::cuda::DeviceGuard guard(devices_[i]);
            EXPECT_EQ(cudaDeviceDisablePeerAccess(devices_[j].index()), cudaErrorPeerAccessNotEnabled)
                << "peer access " << i << " -> " << j << " is still enabled";
            (void)cudaGetLastError();
        }
    }

    tinytorch::cuda::DisableCudaPeerToPeer(devices_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // TearDown disables again
    EXPECT_TRUE(tinytorch::cuda::EnableCudaPeerToPeer(devices_));
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// data[0][e] becomes the sum of data[i][e] over every device i, for every element e.
TEST_P(TinyTorchMultiDeviceReduceParam, ReduceSumToMainUVA)
{
    tinytorch::cuda::AllocatorAlgorithmGuard uva(tinytorch::cuda::AllocatorAlgorithm::CUDA_MALLOC);

    tinytorch::cuda::MultiDeviceTensor mdt(filled(1.0f, devices_[0]), devices_);  // all start at 1.0
    for (size_t i = 1; i < devices_.size(); ++i)
    {
        mdt.data[i] = filled(float(i + 1), devices_[i]);  // device i -> (i + 1)
    }

    mdt.ReduceSumToMainUVA();

    double expected = 0.0;
    for (size_t i = 0; i < devices_.size(); ++i)
    {
        expected += float(i + 1);  // = N * (N + 1) / 2
    }
    TT_EXPECT_CLOSE(mdt.Main(), filled(float(expected), devices_[0]), 0, 0);
}

// Mirrors the real parameter-gradient path that hit the bug: reduce per-device grads to
// the main device's gradient over the full buffer.
TEST_P(TinyTorchMultiDeviceReduceParam, ReduceGradientSumToMainUVA)
{
    tinytorch::cuda::AllocatorAlgorithmGuard uva(tinytorch::cuda::AllocatorAlgorithm::CUDA_MALLOC);

    tinytorch::cuda::MultiDeviceTensor mdt(filled(0.0f, devices_[0]), devices_);
    for (auto& t : mdt.data)
    {
        t.set_requires_grad(true, true);
    }
    for (size_t i = 0; i < devices_.size(); ++i)
    {
        mdt.data[i].mutable_grad() = filled(float(i + 1), devices_[i]);  // device i grad -> (i + 1)
    }

    mdt.ReduceGradientSumToMainUVA();

    double expected = 0.0;
    for (size_t i = 0; i < devices_.size(); ++i)
    {
        expected += float(i + 1);  // = N * (N + 1) / 2
    }
    TT_EXPECT_CLOSE(mdt.data[0].grad(), filled(float(expected), devices_[0]), 0, 0);
}

// The tests below run on a single GPU: both "devices" of the MultiDeviceTensor are cuda:0, so the UVA kernels only
// read local memory.

// A MultiDeviceTensor may hold an empty tensor. The UVA kernels must not be launched with an empty grid (invalid
// configuration error, which release builds leave as the last error).
TEST(TinyTorchMultiDeviceReduce, EmptyTensorUVA)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    tinytorch::cuda::DeviceGuard dg(device);
    (void)cudaGetLastError();
    tinytorch::cuda::AllocatorAlgorithmGuard uva(tinytorch::cuda::AllocatorAlgorithm::CUDA_MALLOC);

    tinytorch::cuda::MultiDeviceTensor mdt(tinytorch::zeros({0}, TensorOptions(device).dtype(kFloat)),
                                           {device, device});
    mdt.ReduceSumToMainUVA();
    mdt.copy_parameters_from_main_to_others_uva();
    TT_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// SetMainAndCopyToOthers accepts an undefined tensor and then clears the others, also with exactly two devices
TEST(TinyTorchMultiDevice, UndefinedMainTwoDevices)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    Device device(kCUDA, 0);
    tinytorch::cuda::DeviceGuard dg(device);
    tinytorch::cuda::MultiDeviceTensor mdt(tinytorch::full({4}, 1.0f, TensorOptions(device).dtype(kFloat)),
                                           {device, device});
    mdt.SetMainAndCopyToOthers(Tensor());
    EXPECT_FALSE(mdt.Main().defined());
    EXPECT_FALSE(mdt.data[1].defined());
}

// The float fast path of ReduceSumCPUSingle reads both tensors linearly, so it must not be used for a non-contiguous
// source (runs on the CPU only)
TEST(TinyTorchMultiDeviceReduce, ReduceSumCPUSingleNonContiguousSource)
{
    tinytorch::cuda::MultiDeviceTensor mdt;
    mdt.cpu_data.resize(2);
    mdt.cpu_data[0] = tinytorch::zeros({2, 2}, TensorOptions(kCPU).dtype(kFloat));
    mdt.cpu_data[1] = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, kCPU, kFloat).permute({1, 0});
    ASSERT_FALSE(mdt.cpu_data[1].is_contiguous());
    mdt.ReduceSumCPUSingle(0, 1);
    TT_EXPECT_CLOSE(mdt.cpu_data[0], tttest::make_tensor({1, 3, 2, 4}, {2, 2}, kCPU, kFloat), 0, 0);
}

#endif  // TT_HAS_CUDA
