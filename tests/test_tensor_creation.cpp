/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <atomic>
#include <cmath>
#include <limits>
#include <thread>

#include "test_utils.h"

using tttest::DeviceTest;

// ----------------------------------------------------------------- empty

TEST(TinyTorchCreate, EmptyProperties)
{
    Tensor t = tinytorch::empty({2, 3});

    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.sizes(), (SizeType{2, 3}));
    EXPECT_EQ(t.dtype(), kFloat);
    EXPECT_EQ(t.device(), kCPU);
    EXPECT_FALSE(t.requires_grad());
}

TEST(TinyTorchCreate, EmptyOptions)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    auto options = TensorOptions().device(kCUDA).dtype(kFloat64).requires_grad(true);

    Tensor t = tinytorch::empty({4}, options);

    EXPECT_EQ(t.device(), kCUDA);
    EXPECT_EQ(t.dtype(), kFloat64);
    EXPECT_TRUE(t.requires_grad());
    EXPECT_EQ(t.element_size(), 8);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchEmptyLike);
TEST_P(TinyTorchEmptyLike, SameShapeAndDevice)
{
    Tensor t = tinytorch::zeros({2, 3}, TensorOptions().device(device()));
    Tensor e = tinytorch::empty_like(t);

    EXPECT_EQ(e.sizes(), t.sizes());
    EXPECT_EQ(e.device(), t.device());
    EXPECT_EQ(e.dtype(), t.dtype());
}

// ----------------------------------------------------------------- zeros/ones/full

TEST(TinyTorchCreate, Zeros)
{
    Tensor t = tinytorch::zeros({2, 3});
    TT_EXPECT_CLOSE(t, tinytorch::zeros({2, 3}), 0.0, 0.0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchZerosLike);
TEST_P(TinyTorchZerosLike, Works)
{
    Tensor t = tinytorch::ones({2, 2}, TensorOptions().device(device()));
    TT_EXPECT_CLOSE(tinytorch::zeros_like(t), tinytorch::zeros({2, 2}, TensorOptions().device(device())), 0.0, 0.0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchOnes);
TEST_P(TinyTorchOnes, Values)
{
    std::vector<float> expected(6, 1.0f);
    TT_EXPECT_CLOSE(tinytorch::ones({2, 3}, TensorOptions().device(device())),
                    tttest::make_tensor(expected, {2, 3}, device()), 0.0, 0.0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchFull);
TEST_P(TinyTorchFull, Values)
{
    std::vector<float> expected(4, 7.5f);
    TT_EXPECT_CLOSE(tinytorch::full({2, 2}, 7.5f, TensorOptions().device(device())),
                    tttest::make_tensor(expected, {2, 2}, device()), 0.0, 0.0);
}

// fill(-0.0) must not take the memset fast path, which produces +0.0
TEST_P(TinyTorchFull, NegativeZeroKeepsSign)
{
    Tensor a = tinytorch::ones({5}, TensorOptions().device(device()));
    tinytorch::fill(a, -0.0);
    Tensor c = a.cpu();
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(c.data_ptr<float>()[i], 0.f);
        EXPECT_TRUE(std::signbit(c.data_ptr<float>()[i])) << i;
    }
}

TEST(TinyTorchCreate, FullLike)
{
    Tensor t = tinytorch::zeros({2, 2});
    std::vector<float> expected(4, 3.0f);
    TT_EXPECT_CLOSE(tinytorch::full_like(t, 3.0f), tttest::make_tensor(expected, {2, 2}, kCPU), 0.0, 0.0);
}

// ----------------------------------------------------------------- random

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchRand);
TEST_P(TinyTorchRand, UniformRangeAndDevice)
{
    Tensor t = tinytorch::rand({64}, TensorOptions().device(device()));

    EXPECT_EQ(t.device(), device());
    EXPECT_EQ(t.dtype(), kFloat);

    auto v = tttest::to_double_vec(t);
    for (double x : v)
    {
        EXPECT_GE(x, 0.0);
        EXPECT_LT(x, 1.0);
    }
}

TEST(TinyTorchCreate, ManualSeedDeterministic)
{
    tinytorch::manual_seed(42);
    Tensor a = tinytorch::rand({8});
    tinytorch::manual_seed(42);
    Tensor b = tinytorch::rand({8});
    Tensor c = tinytorch::rand({8});

    TT_EXPECT_CLOSE(a, b, 0.0, 0.0);
    EXPECT_FALSE(tttest::tensor_near(a, c, 0.0, 0.0));
}

// values are drawn from [low, high) also for half, where rounding to high is common
TEST_P(TinyTorchRand, HalfAndDoubleStayBelowHigh)
{
    for (auto dtype : {tinytorch::kHalf, tinytorch::kDouble})
    {
        Tensor t = tinytorch::rand({200000}, TensorOptions().device(device()).dtype(dtype));
        EXPECT_LT(tinytorch::max(t).toDouble(), 1.0) << dtype;
        EXPECT_GE(tinytorch::min(t).toDouble(), 0.0) << dtype;
        EXPECT_NEAR(t.to(tinytorch::kDouble).mean().toDouble(), 0.5, 0.01) << dtype;
    }
}

TEST_P(TinyTorchRand, ConsecutiveCallsDiffer)
{
    Tensor a = tinytorch::rand({1024}, TensorOptions().device(device()));
    Tensor b = tinytorch::rand({1024}, TensorOptions().device(device()));
    EXPECT_FALSE(tttest::tensor_near(a, b, 0.0, 0.0));
    // no element repeats its value (a zero xorshift state would give constant output)
    EXPECT_GT(tinytorch::std(a).toDouble(), 0.2);
}

// threads that never call manual_seed get distinct streams, also when started at the same time
TEST(TinyTorchCreate, ThreadsGetDistinctStreams)
{
    std::vector<double> r(4);
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
    {
        threads.emplace_back([&r, i]() { r[i] = tinytorch::rand({1}).toDouble(); });
    }
    for (auto& t : threads) t.join();
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
        {
            EXPECT_NE(r[i], r[j]);
        }
    }
}

// manual_seed of one thread also re-seeds the generators of the threads that never called manual_seed
TEST(TinyTorchCreate, ManualSeedReachesOtherThreads)
{
    std::atomic<int> step{0};
    double r1 = 0, r2 = 0;
    std::thread worker(
        [&]()
        {
            tinytorch::rand({1});
            step = 1;
            while (step != 2) std::this_thread::yield();
            r1   = tinytorch::rand({1}).toDouble();
            step = 3;
            while (step != 4) std::this_thread::yield();
            r2 = tinytorch::rand({1}).toDouble();
        });
    while (step != 1) std::this_thread::yield();
    tinytorch::manual_seed(7);
    step = 2;
    while (step != 3) std::this_thread::yield();
    tinytorch::manual_seed(7);
    step = 4;
    worker.join();
    // both draws come from the stream derived from (seed 7, worker thread)
    EXPECT_EQ(r1, r2);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchRandNumerics);
TEST_P(TinyTorchRandNumerics, RandintAllIntegerTypes)
{
    for (auto dtype : {tinytorch::kUInt8, tinytorch::kInt16, tinytorch::kUInt16, tinytorch::kInt32, tinytorch::kLong})
    {
        Tensor t = tinytorch::randint(3, 7, {1000}, TensorOptions().device(device()).dtype(dtype));
        EXPECT_EQ(tinytorch::min(t).toDouble(), 3) << dtype;
        EXPECT_EQ(tinytorch::max(t).toDouble(), 6) << dtype;
    }
    // ranges wider than 32 bit
    Tensor t = tinytorch::randint(0, int64_t(1) << 40, {1000}, TensorOptions().device(device()).dtype(kLong));
    EXPECT_GT(tinytorch::max(t).toDouble(), double(int64_t(1) << 33));
}

TEST(TinyTorchCreate, NumThreads)
{
    tinytorch::set_default_num_threads(3);
    int other = 0;
    std::thread t([&]() { other = tinytorch::get_num_threads(); });
    t.join();
    EXPECT_EQ(other, 3);

    tinytorch::set_num_threads(2);
    EXPECT_EQ(tinytorch::get_num_threads(), 2);
    tinytorch::set_num_threads(0);
    EXPECT_EQ(tinytorch::get_num_threads(), 3);
    tinytorch::set_default_num_threads(0);
    EXPECT_GE(tinytorch::get_num_threads(), 1);
    tinytorch::set_num_threads(-1);
}

// ----------------------------------------------------------------- half

TEST(TinyTorchHalf, FloatConversion)
{
    using tinytorch::Half;
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(float(Half(1.0f)), 1.0f);
    EXPECT_EQ(float(Half(65504.0f)), 65504.0f);
    EXPECT_EQ(float(Half(65520.0f)), inf);
    EXPECT_EQ(float(Half(1e6f)), inf);
    EXPECT_EQ(float(Half(-1e6f)), -inf);
    EXPECT_EQ(float(Half(inf)), inf);
    EXPECT_TRUE(std::isnan(float(Half(std::numeric_limits<float>::quiet_NaN()))));
    // subnormals
    EXPECT_EQ(float(Half(5.9604644775390625e-8f)), 5.9604644775390625e-8f);
    EXPECT_EQ(float(Half(1e-5f)), float(Half::from_bits(168)));
    EXPECT_EQ(float(Half(2.0e-8f)), 0.0f);
    // round to nearest even
    EXPECT_EQ(float(Half(1.0f + 1.0f / 2048)), 1.0f);
    EXPECT_EQ(float(Half(1.0f + 3.0f / 2048)), 1.0f + 2.0f / 1024);
    EXPECT_EQ(float(Half(1.0f + 1.4f / 1024)), 1.0f + 1.0f / 1024);
    EXPECT_EQ(float(Half(-2.5f)), -2.5f);
    // integers convert by value, raw bits only via from_bits
    EXPECT_EQ(float(Half(uint16_t(5))), 5.0f);
    EXPECT_EQ(float(Half(int64_t(-7))), -7.0f);
    EXPECT_EQ(float(Half::from_bits(0x3c00)), 1.0f);

    // exhaustive round trip of all finite halfs
    for (uint32_t b = 0; b < 0x10000; ++b)
    {
        Half h  = Half::from_bits(uint16_t(b));
        float f = float(h);
        if (std::isnan(f)) continue;
        EXPECT_EQ(Half(f).h, uint16_t(b)) << b;
    }
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchHalfConversion);
TEST_P(TinyTorchHalfConversion, TensorRoundTrip)
{
    const double inf = std::numeric_limits<double>::infinity();
    Tensor a         = tttest::make_tensor({1e6, -inf, 1e-5, 0.1}, {4}, device());
    auto v           = tttest::to_double_vec(a.to(tinytorch::kHalf));
    EXPECT_EQ(v[0], inf);
    EXPECT_EQ(v[1], -inf);
    EXPECT_NEAR(v[2], 1e-5, 1e-7);
    EXPECT_EQ(v[3], double(float(tinytorch::Half(0.1f))));
    Tensor b = tttest::make_tensor({1, -2, 300}, {3}, device(), tinytorch::kInt16);
    EXPECT_EQ(tttest::to_double_vec(b.to(tinytorch::kHalf)), (std::vector<double>{1, -2, 300}));
    EXPECT_EQ(tttest::to_double_vec(b.to(tinytorch::kFloat).to(tinytorch::kInt16)), (std::vector<double>{1, -2, 300}));
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchRandint);
TEST_P(TinyTorchRandint, IntRange)
{
    Tensor t = tinytorch::randint(2, 5, {16}, TensorOptions().device(device()).dtype(kLong));

    EXPECT_EQ(t.dtype(), kLong);
    auto v = tttest::to_double_vec(t);
    for (double x : v)
    {
        EXPECT_GE(x, 2);
        EXPECT_LE(x, 5);
        EXPECT_DOUBLE_EQ(std::floor(x), x);
    }
}

// floating point tensors get integer values from [low, high)
TEST_P(TinyTorchRandint, FloatingPointTypes)
{
    for (auto dtype : {kFloat, kFloat64, tinytorch::kHalf})
    {
        Tensor t      = tinytorch::randint(-3, 5, {1000}, TensorOptions().device(device()).dtype(dtype));
        auto v        = tttest::to_double_vec(t);
        bool seen_low = false, seen_high = false;
        for (double x : v)
        {
            EXPECT_EQ(x, std::round(x));
            EXPECT_GE(x, -3);
            EXPECT_LT(x, 5);
            seen_low  = seen_low || x == -3;
            seen_high = seen_high || x == 4;
        }
        EXPECT_TRUE(seen_low) << dtype;
        EXPECT_TRUE(seen_high) << dtype;
    }
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchRandn);
TEST_P(TinyTorchRandn, ApproximatelyStandardNormal)
{
    // Over 65536 samples the sample mean has std 1/sqrt(N) = 1/256 ≈ 0.0039 and the sample
    // std has std-error ~ 1/sqrt(2N) ≈ 0.0028, so the 0.05 tolerance is >12 sigma for both
    // checks (avoids the rare flaky failure of the small-sample 4096 version).
    Tensor t = tinytorch::randn({65536}, TensorOptions().device(device()));
    EXPECT_NEAR(t.mean().toDouble(), 0.0, 0.05);
    EXPECT_NEAR(t.std().toDouble(), 1.0, 0.05);
}

TEST(TinyTorchCreate, RandLikeRandnLike)
{
    Tensor t = tinytorch::zeros({3, 2});
    EXPECT_EQ(tinytorch::rand_like(t).sizes(), (SizeType{3, 2}));
    EXPECT_EQ(tinytorch::randn_like(t).sizes(), (SizeType{3, 2}));
}

// ----------------------------------------------------------------- range

TEST(TinyTorchCreate, Range)
{
    // end is inclusive
    Tensor t                    = tinytorch::range(1, 5, 1);
    std::vector<float> expected = {1, 2, 3, 4, 5};
    TT_EXPECT_CLOSE(t, tttest::make_tensor(expected, {5}, kCPU), 0.0, 0.0);
}

TEST(TinyTorchCreate, RangeNegativeStep)
{
    Tensor t                    = tinytorch::range(3, -1, -2);
    std::vector<float> expected = {3, 1, -1};
    TT_EXPECT_CLOSE(t, tttest::make_tensor(expected, {3}, kCPU), 0.0, 0.0);
}

// the count is robust against floating point error in (end - start) / step
TEST(TinyTorchCreate, RangeFractionalStep)
{
    EXPECT_EQ(tinytorch::range(0, 0.3, 0.1, TensorOptions().dtype(tinytorch::kDouble)).numel(), 4);
    EXPECT_EQ(tinytorch::range(0, 1, 0.1, TensorOptions().dtype(tinytorch::kDouble)).numel(), 11);
    EXPECT_EQ(tinytorch::range(0, 0.35, 0.1, TensorOptions().dtype(tinytorch::kDouble)).numel(), 4);
    EXPECT_EQ(tinytorch::range(1, 0, 1).numel(), 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchFullPrecision);
TEST_P(TinyTorchFullPrecision, LargeIntegerAndDoubleValues)
{
    Tensor a = tinytorch::full({2}, 16777217.0, TensorOptions().device(device()).dtype(kLong));
    EXPECT_EQ(tttest::to_double_vec(a), (std::vector<double>{16777217, 16777217}));
    Tensor b = tinytorch::full({1}, 0.1, TensorOptions().device(device()).dtype(tinytorch::kDouble));
    EXPECT_EQ(b.toDouble(), 0.1);
}

// ----------------------------------------------------------------- from_blob

TEST(TinyTorchCreate, FromBlobFloat)
{
    std::vector<float> data = {1, 2, 3, 4};
    Tensor t                = tinytorch::from_blob(data.data(), {2, 2});

    EXPECT_EQ(t.sizes(), (SizeType{2, 2}));
    EXPECT_EQ(t.dtype(), kFloat);
    TT_EXPECT_CLOSE(t, tttest::make_tensor(data, {2, 2}, kCPU), 0.0, 0.0);
}

TEST(TinyTorchCreate, FromBlobDtype)
{
    std::vector<int64_t> data = {7, 8, 9};
    Tensor t                  = tinytorch::from_blob(data.data(), {3}, kLong);

    EXPECT_EQ(t.dtype(), kLong);
    TT_EXPECT_CLOSE(t, tttest::make_tensor(data, {3}, kCPU), 0.0, 0.0);
}

TEST(TinyTorchCreate, FromBlobCustomStrides)
{
    // 2x2 tensor reading a 2x3 buffer with row stride 3 (non-contiguous):
    // t[i][j] = data[i * 3 + j]
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t                = tinytorch::from_blob(data.data(), {2, 2}, SizeType{3, 1}, kFloat);

    EXPECT_FALSE(t.is_contiguous());
    std::vector<float> expected = {1, 2, 4, 5};
    TT_EXPECT_CLOSE(t, tttest::make_tensor(expected, {2, 2}, kCPU), 0.0, 0.0);
}

TEST(TinyTorchCreate, FromBlobDoesNotTakeOwnership)
{
    // The tensor must still be valid after this scope: the buffer stays alive in `data`.
    std::vector<float> data     = {1, 2, 3, 4, 5, 6};
    Tensor t                    = tinytorch::from_blob(data.data(), {2, 3});
    std::vector<float> expected = {1, 2, 3, 4, 5, 6};
    TT_EXPECT_CLOSE(t, tttest::make_tensor(expected, {2, 3}, kCPU), 0.0, 0.0);
}
