/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

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

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchRandn);
TEST_P(TinyTorchRandn, ApproximatelyStandardNormal)
{
    // zero-mean check (loose): over 4096 samples the mean is ~ N(0, 1/64)
    Tensor t = tinytorch::randn({4096}, TensorOptions().device(device()));
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
    Tensor t = tinytorch::range(1, 5, 1);
    std::vector<float> expected = {1, 2, 3, 4, 5};
    TT_EXPECT_CLOSE(t, tttest::make_tensor(expected, {5}, kCPU), 0.0, 0.0);
}

TEST(TinyTorchCreate, RangeNegativeStep)
{
    Tensor t = tinytorch::range(3, -1, -2);
    std::vector<float> expected = {3, 1, -1};
    TT_EXPECT_CLOSE(t, tttest::make_tensor(expected, {3}, kCPU), 0.0, 0.0);
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
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor t                = tinytorch::from_blob(data.data(), {2, 3});
    std::vector<float> expected = {1, 2, 3, 4, 5, 6};
    TT_EXPECT_CLOSE(t, tttest::make_tensor(expected, {2, 3}, kCPU), 0.0, 0.0);
}
