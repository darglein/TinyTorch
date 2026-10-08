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
    int maxc   = std::min(avail, 4);
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

    static Tensor filled(float v, Device d)
    {
        return tinytorch::full({kRows, kCols}, v, TensorOptions().device(d));
    }
};

INSTANTIATE_TEST_SUITE_P(DeviceCounts, TinyTorchMultiDeviceReduceParam, ::testing::ValuesIn(reduce_device_counts()),
                         [](const ::testing::TestParamInfo<int>& info)
                         {
                             return "devices" + std::to_string(info.param);
                         });

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

#endif  // TT_HAS_CUDA
