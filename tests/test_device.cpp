/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

namespace
{

// Returns a small 2x3 tensor of known values on the given device.
Tensor base_tensor(Device d)
{
    return tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, d);
}

}  // namespace

// ------------------------------------------------------------------ device transfer

TEST(TinyTorchDevice, ToDeviceRoundTrip)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);

    Tensor c = a.to(kCUDA);
    EXPECT_EQ(c.device(), kCUDA);
    TT_EXPECT_CLOSE(c, a, 0, 0);

    Tensor back = c.to(kCPU);
    EXPECT_EQ(back.device(), kCPU);
    TT_EXPECT_CLOSE(back, a, 0, 0);
}

TEST(TinyTorchDevice, CudaCpuMethods)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);
    Tensor c = a.cuda();
    EXPECT_TRUE(c.is_cuda());
    EXPECT_FALSE(c.is_cpu());

    Tensor b = c.cpu();
    EXPECT_TRUE(b.is_cpu());
    TT_EXPECT_CLOSE(b, a, 0, 0);
}

TEST(TinyTorchDevice, ToInPlace)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);
    a.to_(kCUDA);
    EXPECT_EQ(a.device(), kCUDA);
    TT_EXPECT_CLOSE(a, base_tensor(kCPU), 0, 0);
    a.to_(kCPU);
    EXPECT_EQ(a.device(), kCPU);
}

TEST(TinyTorchDevice, CopyCrossDevice)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);
    Tensor c = tinytorch::empty({2, 3}, TensorOptions().device(kCUDA));

    tinytorch::copy(a, c, false);
    TT_EXPECT_CLOSE(c, a, 0, 0);

    tinytorch::copy(c, a, false);
    TT_EXPECT_CLOSE(a, base_tensor(kCPU), 0, 0);
}

TEST(TinyTorchDevice, DtypeConversionCrossDevice)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU).to(kFloat64);
    Tensor c = a.to(kCUDA);

    EXPECT_EQ(c.device(), kCUDA);
    EXPECT_EQ(c.dtype(), kFloat64);
    TT_EXPECT_CLOSE(c, a, 1e-6, 1e-6);
}

// ------------------------------------------------------------------ autograd across devices

TEST(TinyTorchDevice, AutogradOnCuda)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = tttest::leaf({1, 2, 3, 4}, {2, 2}, kCUDA);
    Tensor b = tttest::leaf({10, 20, 30, 40}, {2, 2}, kCUDA);

    Tensor r = a * b;
    tinytorch::backward(tinytorch::sum(r));

    TT_EXPECT_CLOSE(a.grad(), b, 0, 0);
    TT_EXPECT_CLOSE(b.grad(), a, 0, 0);
}

TEST(TinyTorchDevice, AutogradWithDeviceMove)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    // move a leaf to the GPU, then build the graph there
    Tensor a = tttest::leaf({1, 2, 3, 4}, {4}, kCPU);
    a.to_(kCUDA);

    Tensor r = a * a;
    tinytorch::backward(tinytorch::sum(r));
    // d(a^2)/da = 2a
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({2, 4, 6, 8}, {4}, kCUDA), 0, 0);
}

// ------------------------------------------------------------------ CPU/CUDA agreement

// Runs the same computation on both devices and compares the results.
class TinyTorchDeviceAgreement : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        if (!tttest::has_cuda())
        {
            GTEST_SKIP() << "no CUDA device available";
        }
    }

    void expect_same(const Tensor& cpu_result, const Tensor& cuda_result, double rtol = 1e-4, double atol = 1e-5)
    {
        TT_EXPECT_CLOSE(cuda_result, cpu_result, rtol, atol);
    }
};

TEST_F(TinyTorchDeviceAgreement, ElementwiseOpsMatch)
{
    Tensor a_cpu = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, kCPU);
    Tensor b_cpu = tttest::make_tensor({5, 6, 7, 8}, {2, 2}, kCPU);

    Tensor a_g = a_cpu.to(kCUDA);
    Tensor b_g = b_cpu.to(kCUDA);

    expect_same(a_cpu + b_cpu, a_g + b_g);
    expect_same(a_cpu - b_cpu, a_g - b_g);
    expect_same(a_cpu * b_cpu, a_g * b_g);
    expect_same(a_cpu / b_cpu, a_g / b_g);
    expect_same(tinytorch::sqrt(a_cpu), tinytorch::sqrt(a_g));
    expect_same(tinytorch::exp(a_cpu), tinytorch::exp(a_g));
    expect_same(tinytorch::relu(a_cpu - 3.0), tinytorch::relu(a_g - 3.0));
    expect_same(tinytorch::sigmoid(a_cpu), tinytorch::sigmoid(a_g));
    expect_same(tinytorch::sum(a_cpu), tinytorch::sum(a_g));
    expect_same(tinytorch::mean(a_cpu), tinytorch::mean(a_g));
    expect_same(tinytorch::cumsum(a_cpu, 1), tinytorch::cumsum(a_g, 1));
    expect_same(tinytorch::matmul(a_cpu, b_cpu), tinytorch::matmul(a_g, b_g));
    expect_same(tinytorch::permute(a_cpu, {1, 0}), tinytorch::permute(a_g, {1, 0}));
    expect_same(tinytorch::slice(a_cpu, 0, 1, 2, 1), tinytorch::slice(a_g, 0, 1, 2, 1));
}

// ------------------------------------------------------------------ MultiDeviceTensor (CUDA only)

#ifdef TT_HAS_CUDA
TEST(TinyTorchMultiDevice, SingleDeviceBasics)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor main = base_tensor(kCUDA);
    tinytorch::cuda::MultiDeviceTensor mdt(main, std::vector<Device>{kCUDA});

    EXPECT_EQ(mdt.size(), 1);
    EXPECT_TRUE(mdt.Initialized());
    EXPECT_TRUE(mdt.defined());
    TT_EXPECT_CLOSE(mdt.Main(), main, 0, 0);
    TT_EXPECT_CLOSE(mdt[Device(kCUDA, 0)], main, 0, 0);

    mdt.zero_();
    TT_EXPECT_CLOSE(mdt.Main(), tinytorch::zeros({2, 3}, TensorOptions().device(kCUDA)), 0, 0);
}
#endif  // TT_HAS_CUDA
