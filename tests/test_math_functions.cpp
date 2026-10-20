/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <cmath>
#include <limits>

#include "test_utils.h"

using tttest::DeviceTest;

namespace
{

// Folds in the cases from the former tests/test_backward.cpp.

// d(sum(a))/da = 1 for every element
TEST(TinyTorchBackward, Sum)
{
    Tensor a = tinytorch::zeros({2, 3});
    a.set_requires_grad(true, true);

    tinytorch::backward(tinytorch::sum(a));

    Tensor expected = tinytorch::ones({2, 3});
    EXPECT_TRUE(a.grad().allclose(expected));
}

// d(mean(a))/da = 1 / numel(a)
TEST(TinyTorchBackward, Mean)
{
    Tensor a = tinytorch::zeros({2, 3});
    a.set_requires_grad(true, true);

    tinytorch::backward(tinytorch::mean(a));

    Tensor expected = tinytorch::full({2, 3}, 1.0f / 6.0f);
    EXPECT_TRUE(a.grad().allclose(expected));
}

}  // namespace

// ------------------------------------------------------------------ sum

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSumOps);
TEST_P(TinyTorchSumOps, Global)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    TT_EXPECT_CLOSE(tinytorch::sum(a), tttest::make_tensor({21}, {1}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSumDim);
TEST_P(TinyTorchSumDim, WithAndWithoutKeepDim)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor s0 = tinytorch::sum(a, 0, true);
    TT_EXPECT_CLOSE(s0, tttest::make_tensor({5, 7, 9}, {1, 3}, device()), 1e-5, 1e-6);

    Tensor s0s = tinytorch::sum(a, 0, false);
    EXPECT_EQ(s0s.sizes(), (SizeType{3}));

    Tensor s1 = tinytorch::sum(a, 1, true);
    TT_EXPECT_CLOSE(s1, tttest::make_tensor({6, 15}, {2, 1}, device()), 1e-5, 1e-6);

    // size-type variant
    Tensor s12 = tinytorch::sum(a, SizeType{0, 1}, true);
    TT_EXPECT_CLOSE(s12, tttest::make_tensor({21}, {1, 1}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSumGradient);
TEST_P(TinyTorchSumGradient, GlobalAndDim)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    a.set_requires_grad(true, true);

    tinytorch::backward(tinytorch::sum(a));
    TT_EXPECT_CLOSE(a.grad(), tinytorch::ones({2, 2}, TensorOptions().device(device())), 0, 0);

    // backward() requires a scalar loss, so reduce the dim-sum again
    a.mutable_grad().zero_();
    tinytorch::backward(tinytorch::sum(tinytorch::sum(a, 0, true)));
    TT_EXPECT_CLOSE(a.grad(), tinytorch::ones({2, 2}, TensorOptions().device(device())), 0, 0);

    a.mutable_grad().zero_();
    tinytorch::backward(tinytorch::sum(tinytorch::sum(a, 1, true)));
    TT_EXPECT_CLOSE(a.grad(), tinytorch::ones({2, 2}, TensorOptions().device(device())), 0, 0);
}

// ------------------------------------------------------------------ mean

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMeanOps);
TEST_P(TinyTorchMeanOps, GlobalAndDim)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    // a = [[1,2],[3,4]]: column means [2,3], row means [1.5,3.5]
    TT_EXPECT_CLOSE(tinytorch::mean(a), tttest::make_tensor({2.5}, {1}, device()), 1e-5, 1e-6);
    TT_EXPECT_CLOSE(tinytorch::mean(a, 0, true), tttest::make_tensor({2, 3}, {1, 2}, device()), 1e-5, 1e-6);
    TT_EXPECT_CLOSE(tinytorch::mean(a, 1, true), tttest::make_tensor({1.5, 3.5}, {2, 1}, device()), 1e-5, 1e-6);
}

// ------------------------------------------------------------------ min/max

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMinMaxBinary);
TEST_P(TinyTorchMinMaxBinary, Values)
{
    Tensor a = tttest::make_tensor({1, 5, 3, 4}, {2, 2}, device());
    Tensor b = tttest::make_tensor({2, 4, 3, 9}, {2, 2}, device());

    TT_EXPECT_CLOSE(tinytorch::min(a, b), tttest::make_tensor({1, 4, 3, 4}, {2, 2}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::max(a, b), tttest::make_tensor({2, 5, 3, 9}, {2, 2}, device()), 0, 0);
}

// both operands broadcast, also the first one
TEST_P(TinyTorchMinMaxBinary, BroadcastBothOperands)
{
    Tensor a = tttest::make_tensor({2, 2, 2}, {1, 3}, device());
    Tensor b = tttest::make_tensor({1, 2, 3, 4, 0, 6}, {2, 3}, device());
    TT_EXPECT_CLOSE(tinytorch::min(a, b), tttest::make_tensor({1, 2, 2, 2, 0, 2}, {2, 3}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::max(a, b), tttest::make_tensor({2, 2, 3, 4, 2, 6}, {2, 3}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::max(b, tttest::make_tensor({2}, {1}, device())),
                    tttest::make_tensor({2, 2, 3, 4, 2, 6}, {2, 3}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMinMaxReduce);
TEST_P(TinyTorchMinMaxReduce, ValuesAndIndices)
{
    Tensor a = tttest::make_tensor({3, 1, 4, 1, 5, 9}, {2, 3}, device());

    // a = [[3,1,4],[1,5,9]]; reducing dim 1 with keepdim gives shape {2,1};
    // indices are 0-based within the reduced dim
    auto [mn, mn_idx] = tinytorch::min(a, 1, true);
    TT_EXPECT_CLOSE(mn, tttest::make_tensor({1, 1}, {2, 1}, device()), 0, 0);

    auto [mx, mx_idx] = tinytorch::max(a, 1, true);
    TT_EXPECT_CLOSE(mx, tttest::make_tensor({4, 9}, {2, 1}, device()), 0, 0);

    TT_EXPECT_CLOSE(mn_idx, tttest::make_tensor(std::vector<int64_t>{1, 0}, {2, 1}, device()), 0, 0);
    TT_EXPECT_CLOSE(mx_idx, tttest::make_tensor(std::vector<int64_t>{2, 2}, {2, 1}, device()), 0, 0);

    // without keepdim
    auto [mns, mns_idx] = tinytorch::min(a, 1, false);
    EXPECT_EQ(mns.sizes(), (SizeType{2}));
    EXPECT_EQ(mns_idx.sizes(), (SizeType{2}));

    // global
    TT_EXPECT_CLOSE(tinytorch::min(a), tttest::make_tensor({1}, {1}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::max(a), tttest::make_tensor({9}, {1}, device()), 0, 0);
}

// ------------------------------------------------------------------ prod/cumsum/cumprod

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchProdOps);
TEST_P(TinyTorchProdOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor p = tinytorch::prod(a, 1, true);
    TT_EXPECT_CLOSE(p, tttest::make_tensor({6, 120}, {2, 1}, device()), 1e-5, 1e-6);

    Tensor p0 = tinytorch::prod(a, 0, true);
    TT_EXPECT_CLOSE(p0, tttest::make_tensor({4, 10, 18}, {1, 3}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchCumSum);
TEST_P(TinyTorchCumSum, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    TT_EXPECT_CLOSE(tinytorch::cumsum(a, 0), tttest::make_tensor({1, 2, 4, 6}, {2, 2}, device()), 1e-5, 1e-6);
    TT_EXPECT_CLOSE(tinytorch::cumsum(a, 1), tttest::make_tensor({1, 3, 3, 7}, {2, 2}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchCumProd);
TEST_P(TinyTorchCumProd, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    TT_EXPECT_CLOSE(tinytorch::cumprod(a, 1), tttest::make_tensor({1, 2, 3, 12}, {2, 2}, device()), 1e-5, 1e-6);
}

// ------------------------------------------------------------------ reduction regressions

TEST_P(TinyTorchMinMaxReduce, LargeMagnitudes)
{
    Tensor a = tttest::make_tensor({2e6, 3e6, 5e6}, {3}, device());
    EXPECT_EQ(tinytorch::min(a).toDouble(), 2e6);
    Tensor b = tttest::make_tensor({-2e6, -3e6, -5e6}, {3}, device());
    EXPECT_EQ(tinytorch::max(b).toDouble(), -2e6);

    Tensor c          = tttest::make_tensor({2e6, 3e6, -3e6, -2e6}, {2, 2}, device());
    auto [mn, mn_idx] = tinytorch::min(c, 1, false);
    auto [mx, mx_idx] = tinytorch::max(c, 1, false);
    EXPECT_EQ(tttest::to_double_vec(mn), (std::vector<double>{2e6, -3e6}));
    EXPECT_EQ(tttest::to_double_vec(mx), (std::vector<double>{3e6, -2e6}));
    EXPECT_EQ(tttest::to_double_vec(mn_idx), (std::vector<double>{0, 0}));
    EXPECT_EQ(tttest::to_double_vec(mx_idx), (std::vector<double>{1, 1}));
}

// a NaN in the slice is the result (like PyTorch), and the index always points to it
TEST_P(TinyTorchMinMaxReduce, NaNPropagates)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (auto dtype : {tinytorch::kHalf, tinytorch::kFloat, tinytorch::kDouble})
    {
        Tensor a          = tttest::make_tensor({1, nan, 0, nan, 2, 3}, {2, 3}, device(), dtype);
        auto [mn, mn_idx] = tinytorch::min(a, 1, false);
        auto [mx, mx_idx] = tinytorch::max(a, 1, false);
        auto mnv          = tttest::to_double_vec(mn);
        auto mxv          = tttest::to_double_vec(mx);
        EXPECT_TRUE(std::isnan(mnv[0]) && std::isnan(mnv[1])) << dtype;
        EXPECT_TRUE(std::isnan(mxv[0]) && std::isnan(mxv[1])) << dtype;
        EXPECT_EQ(tttest::to_double_vec(mn_idx), (std::vector<double>{1, 0})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(mx_idx), (std::vector<double>{1, 0})) << dtype;

        EXPECT_TRUE(std::isnan(tinytorch::min(a).toDouble())) << dtype;
        EXPECT_TRUE(std::isnan(tinytorch::max(a).toDouble())) << dtype;
    }
}

TEST_P(TinyTorchMinMaxReduce, IntegerTypes)
{
    for (auto dtype : {tinytorch::kUInt8, tinytorch::kInt16, tinytorch::kUInt16, tinytorch::kInt32, tinytorch::kLong})
    {
        Tensor a          = tttest::make_tensor({5, 3, 7, 3, 200, 1}, {2, 3}, device(), dtype);
        auto [mn, mn_idx] = tinytorch::min(a, -1, false);
        auto [mx, mx_idx] = tinytorch::max(a, -1, false);
        EXPECT_EQ(tttest::to_double_vec(mn), (std::vector<double>{3, 1})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(mx), (std::vector<double>{7, 200})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(mn_idx), (std::vector<double>{1, 2})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(mx_idx), (std::vector<double>{2, 1})) << dtype;
        EXPECT_EQ(tinytorch::min(a).toDouble(), 1) << dtype;
        EXPECT_EQ(tinytorch::max(a).toDouble(), 200) << dtype;
    }
    Tensor b          = tttest::make_tensor({-5, 3, -7}, {3}, device(), tinytorch::kInt32);
    auto [mn, mn_idx] = tinytorch::min(b, 0, false);
    EXPECT_EQ(mn.toDouble(), -7);
    EXPECT_EQ(mn_idx.toDouble(), 2);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchNegativeDim);
TEST_P(TinyTorchNegativeDim, Reductions)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    TT_EXPECT_CLOSE(tinytorch::sum(a, -1, false), tinytorch::sum(a, 1, false), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::sum(a, -2, true), tinytorch::sum(a, 0, true), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::mean(a, -1, false), tinytorch::mean(a, 1, false), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::sum(a, SizeType{-1, -2}, false), tinytorch::sum(a), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::max(a, -1, false).first, tinytorch::max(a, 1, false).first, 0, 0);
    TT_EXPECT_CLOSE(tinytorch::max(a, -1, false).second, tinytorch::max(a, 1, false).second, 0, 0);
    TT_EXPECT_CLOSE(tinytorch::min(a, -1, false).first, tinytorch::min(a, 1, false).first, 0, 0);
    TT_EXPECT_CLOSE(tinytorch::prod(a, -1, false), tinytorch::prod(a, 1, false), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::cumsum(a, -1), tinytorch::cumsum(a, 1), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::cumprod(a, -1), tinytorch::cumprod(a, 1), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::std(a, -1), tinytorch::std(a, 1), 0, 0);
}

// half and integer sums don't accumulate in their own type
TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSumAccumulation);
TEST_P(TinyTorchSumAccumulation, HalfAccumulatesInFloat)
{
    Tensor a = tinytorch::ones({2, 4096}, TensorOptions().device(device()).dtype(tinytorch::kHalf));
    EXPECT_EQ(tinytorch::sum(a).toDouble(), 8192);
    EXPECT_EQ(tttest::to_double_vec(tinytorch::sum(a, 1, false)), (std::vector<double>{4096, 4096}));
    Tensor c = tinytorch::cumsum(a, 1);
    EXPECT_EQ(tttest::to_double_vec(c.slice(1, 4095, 4096)), (std::vector<double>{4096, 4096}));
    EXPECT_EQ(tinytorch::abs_sum(a).toDouble(), 8192);
}

TEST_P(TinyTorchSumAccumulation, IntegerTypes)
{
    for (auto dtype : {tinytorch::kUInt8, tinytorch::kInt16, tinytorch::kUInt16, tinytorch::kInt32, tinytorch::kLong})
    {
        Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device(), dtype);
        EXPECT_EQ(tinytorch::sum(a).toDouble(), 21) << dtype;
        EXPECT_EQ(tttest::to_double_vec(tinytorch::sum(a, 1, false)), (std::vector<double>{6, 15})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(tinytorch::sum(a, 0, false)), (std::vector<double>{5, 7, 9})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(tinytorch::cumsum(a, 1)), (std::vector<double>{1, 3, 6, 4, 9, 15})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(tinytorch::cumprod(a, 1)), (std::vector<double>{1, 2, 6, 4, 20, 120})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(tinytorch::prod(a, 1, false)), (std::vector<double>{6, 120})) << dtype;
    }
    // uint16 values summed exactly (a float accumulator would round)
    Tensor b = tinytorch::full({20000}, 65535, TensorOptions().device(device()).dtype(tinytorch::kLong));
    EXPECT_EQ(tinytorch::sum(b).toDouble(), 20000.0 * 65535.0);
}

// compares the dim reductions against a double reference for short (one thread per output) and long reductions
TEST_P(TinyTorchSumAccumulation, DimReduceShapes)
{
    for (SizeType shape : {SizeType{3, 5}, SizeType{1000, 7}, SizeType{7, 1000}, SizeType{40, 33}, SizeType{2, 3, 300}})
    {
        Tensor a   = tinytorch::rand(shape, TensorOptions().device(device()));
        Tensor ref = a.cpu().to(tinytorch::kDouble);
        for (int64_t d = 0; d < a.dim(); ++d)
        {
            TT_EXPECT_CLOSE(tinytorch::sum(a, d, false).to(tinytorch::kDouble),
                            tinytorch::sum(ref, d, false).to(a.device()), 1e-5, 1e-5);
            TT_EXPECT_CLOSE(tinytorch::cumsum(a, d).to(tinytorch::kDouble), tinytorch::cumsum(ref, d).to(a.device()),
                            1e-5, 1e-5);
        }
    }
}

// sum over an empty dim gives 0 (and the size-1 result, not the empty input)
TEST_P(TinyTorchSumAccumulation, EmptyDim)
{
    Tensor a = tinytorch::zeros({0, 3}, TensorOptions().device(device()));
    Tensor s = tinytorch::sum(a, 0, true);
    EXPECT_EQ(s.sizes(), (SizeType{1, 3}));
    EXPECT_EQ(tttest::to_double_vec(s), (std::vector<double>{0, 0, 0}));
    EXPECT_EQ(tinytorch::cumsum(a, 0).sizes(), (SizeType{0, 3}));
}

// ------------------------------------------------------------------ pow

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPowScalar);
TEST_P(TinyTorchPowScalar, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 2, 3, 4}, {2, 2}, device());

    Tensor r = tinytorch::pow(a, 2.0);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({1, 4, 9, 16}, {2, 2}, device()), 1e-5, 1e-6);

    // d(a^b)/da = b * a^(b-1)
    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({2, 4, 6, 8}, {2, 2}, device()), 1e-4, 1e-6);
}

// pow(x, 0) has the gradient 0 everywhere (PyTorch), also at x = 0 where pow(x, -1) is inf
TEST_P(TinyTorchPowScalar, ZeroExponentAtZero)
{
    Tensor a = tttest::leaf({0, 2}, {2}, device());
    tinytorch::backward(tinytorch::sum(tinytorch::pow(a, 0.0)));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({0, 0}, {2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPowTensor);
TEST_P(TinyTorchPowTensor, ValuesAndGradient)
{
    Tensor a = tttest::leaf({2, 3}, {2}, device());
    Tensor b = tttest::leaf({2, 3}, {2}, device());

    Tensor r = tinytorch::pow(a, b);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({4, 27}, {2}, device()), 1e-5, 1e-6);

    tttest::check_grads(
        {a, b}, [](const std::vector<Tensor>& p) { return tinytorch::sum(tinytorch::pow(p[0], p[1])); }, 2e-2, 2e-3);
}

// d/da a^b at a = 0, b = 0 is 0 and d/db a^b at a = 0, b > 0 is 0 (PyTorch masks both)
TEST_P(TinyTorchPowTensor, GradientAtZeroBase)
{
    Tensor a = tttest::leaf({0, 0, 2}, {3}, device());
    Tensor b = tttest::leaf({0, 2, 3}, {3}, device());
    tinytorch::backward(tinytorch::sum(tinytorch::pow(a, b)));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({0, 0, 12}, {3}, device()), 1e-5, 1e-5);
    TT_EXPECT_CLOSE(b.grad(), tttest::make_tensor({0, 0, 8 * std::log(2.0)}, {3}, device()), 1e-5, 1e-5);
}

// integer pow is exact, also on the GPU (3^1 is 3, not 2 from a single precision pow)
TEST_P(TinyTorchPowTensor, IntegerTypesExact)
{
    tinytorch::NoGradGuard ngg;
    for (auto t : {tinytorch::kUInt8, tinytorch::kInt16, tinytorch::kInt32, tinytorch::kLong})
    {
        auto expected = [&](const std::vector<int64_t>& v) { return tttest::make_tensor(v, {6}, device()).to(t); };
        Tensor a      = expected({3, 5, 6, 2, 3, 4});
        Tensor b      = expected({1, 3, 2, 5, 3, 2});
        TT_EXPECT_CLOSE(tinytorch::pow(a, b), expected({3, 125, 36, 32, 27, 16}), 0, 0);
        TT_EXPECT_CLOSE(tinytorch::pow(a, 1.0), a, 0, 0);
        TT_EXPECT_CLOSE(tinytorch::pow(a, 3.0), expected({27, 125, 216, 8, 27, 64}), 0, 0);
        TT_EXPECT_CLOSE(tinytorch::pow(a, 0.5), expected({1, 2, 2, 1, 1, 2}), 0, 0);
    }
}

// ------------------------------------------------------------------ clamp

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchClampOps);
TEST_P(TinyTorchClampOps, Values)
{
    Tensor a = tttest::make_tensor({-5, -1, 0, 1, 5}, {1, 5}, device());

    TT_EXPECT_CLOSE(tinytorch::clamp(a, -1, 1), tttest::make_tensor({-1, -1, 0, 1, 1}, {1, 5}, device()), 0, 0);

    Tensor b = tttest::make_tensor({-5, -1, 0, 1, 5}, {1, 5}, device());
    tinytorch::clamp_(b, -2, 2);
    TT_EXPECT_CLOSE(b, tttest::make_tensor({-2, -1, 0, 1, 2}, {1, 5}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchTensorClampMinMethods);
TEST_P(TinyTorchTensorClampMinMethods, Values)
{
    Tensor a = tttest::make_tensor({-5, 0, 5}, {3}, device());

    Tensor c = a.clamp_min(-1);
    TT_EXPECT_CLOSE(c, tttest::make_tensor({-1, 0, 5}, {3}, device()), 0, 0);

    a.clamp_max_(2);
    TT_EXPECT_CLOSE(a, tttest::make_tensor({-5, 0, 2}, {3}, device()), 0, 0);
}

// ------------------------------------------------------------------ norm/std

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchNormOps);
TEST_P(TinyTorchNormOps, FrobeniusAlongDim)
{
    Tensor a = tttest::make_tensor({3, 4, 5, 12}, {2, 2}, device());

    // only norm == 2 is implemented
    Tensor n = tinytorch::norm(a, 2, 1, true);
    TT_EXPECT_CLOSE(n, tttest::make_tensor({5, 13}, {2, 1}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchStdOps);
TEST_P(TinyTorchStdOps, Global)
{
    // values {1, 2, 3, 4}: variance (population) = 1.25, std = sqrt(1.25)
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device());
    TT_EXPECT_CLOSE(tinytorch::std(a), tttest::make_tensor({std::sqrt(1.25)}, {1}, device()), 1e-4, 1e-6);
}

TEST_P(TinyTorchStdOps, PerDim)
{
    // a = [[1,2,3],[4,5,6]]
    // per-column (dim 0): means {2.5,3.5,4.5}, deviations +-1.5 -> std = 1.5 each
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    TT_EXPECT_CLOSE(tinytorch::std(a, 0), tttest::make_tensor({1.5, 1.5, 1.5}, {3}, device()), 1e-4, 1e-6);

    // per-row (dim 1): mean 2 (resp. 5), deviations {+-1,0} -> var 2/3
    TT_EXPECT_CLOSE(tinytorch::std(a, 1), tttest::make_tensor({std::sqrt(2.0 / 3), std::sqrt(2.0 / 3)}, {2}, device()),
                    1e-4, 1e-6);

    // gradient along a negative dim
    Tensor b =
        tttest::leaf({0.3f, -1.2f, 0.7f, 1.5f, -0.4f, 0.9f, 0.1f, 0.2f, -0.6f, 1.1f, 0.8f, -0.9f}, {3, 4}, device());
    tttest::check_grads({b}, [](const std::vector<Tensor>& p) { return tinytorch::sum(tinytorch::std(p[0], -1)); });
}

// ------------------------------------------------------------------ abs_sum / prod_sum

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchAbsSum);
TEST_P(TinyTorchAbsSum, Values)
{
    Tensor a = tttest::make_tensor({-1, 2, -3, 4}, {2, 2}, device());
    TT_EXPECT_CLOSE(tinytorch::abs_sum(a), tttest::make_tensor({10}, {1}, device()), 1e-5, 1e-6);
}

TEST_P(TinyTorchAbsSum, Backward)
{
    // d/dx sum(|x|) = sign(x); avoid the kinks at 0
    Tensor a = tttest::leaf({1.0f, -2.0f, 3.0f, -4.0f}, {4}, device());
    tttest::check_grads({a}, [](const std::vector<Tensor>& p) { return tinytorch::abs_sum(p[0]); }, 2e-2, 2e-3);
}

// "prod_sum" is the sum of squares: sum(v^2)
TT_INSTANTIATE_DEVICE_TESTS(TinyTorchProdSum);
TEST_P(TinyTorchProdSum, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    TT_EXPECT_CLOSE(tinytorch::prod_sum(a), tttest::make_tensor({30}, {1}, device()), 1e-5, 1e-6);
}

TEST_P(TinyTorchProdSum, Backward)
{
    // d/dx sum(x^2) = 2x
    Tensor a = tttest::leaf({1.0f, -2.0f, 3.0f, -4.0f}, {4}, device());
    tttest::check_grads({a}, [](const std::vector<Tensor>& p) { return tinytorch::prod_sum(p[0]); }, 2e-2, 2e-3);
}

// ------------------------------------------------------------------ median

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMedian);
TEST_P(TinyTorchMedian, Values)
{
    Tensor a = tttest::make_tensor({3, 1, 2}, {3}, device());
    TT_EXPECT_CLOSE(tinytorch::median(a, 0.5), tttest::make_tensor({2}, {1}, device()), 1e-5, 1e-6);
}

// like torch.median: the lower of the two middle values for an even count
TEST_P(TinyTorchMedian, EvenCountReturnsLowerMedian)
{
    Tensor a = tttest::make_tensor({4, 1, 3, 2}, {4}, device());
    EXPECT_EQ(tinytorch::median(a, 0.5).toDouble(), 2);
}

TEST_P(TinyTorchMedian, PercentileBoundsAndNaN)
{
    Tensor a = tttest::make_tensor({4, 1, 3, 2}, {4}, device());
    EXPECT_EQ(tinytorch::median(a, 1.0).toDouble(), 4);
    EXPECT_EQ(tinytorch::median(a, 0.0).toDouble(), 1);
    EXPECT_EQ(tinytorch::median(a, 0.75).toDouble(), 3);

    Tensor b = tttest::make_tensor({1, std::numeric_limits<double>::quiet_NaN(), 3}, {3}, device());
    EXPECT_TRUE(std::isnan(tinytorch::median(b, 0.5).toDouble()));
}

// ------------------------------------------------------------------ sort

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSortNumerics);
TEST_P(TinyTorchSortNumerics, NaNSortsLast)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Tensor a         = tttest::make_tensor({3, nan, 1, nan, 2}, {5}, device());
    auto [v, idx]    = tinytorch::sort(a, 0);
    auto vv          = tttest::to_double_vec(v);
    EXPECT_EQ(std::vector<double>(vv.begin(), vv.begin() + 3), (std::vector<double>{1, 2, 3}));
    EXPECT_TRUE(std::isnan(vv[3]) && std::isnan(vv[4]));
    EXPECT_EQ(tttest::to_double_vec(idx), (std::vector<double>{2, 4, 0, 1, 3}));
}

TEST_P(TinyTorchSortNumerics, EmptyDim)
{
    Tensor a      = tinytorch::zeros({2, 0}, TensorOptions().device(device()));
    auto [v, idx] = tinytorch::sort(a, 1);
    EXPECT_EQ(v.sizes(), (SizeType{2, 0}));
    EXPECT_EQ(idx.sizes(), (SizeType{2, 0}));
}

// ------------------------------------------------------------------ clamp / poisson numerics

// bounds outside the range of an integer type saturate instead of wrapping (uint8(300) would be 44)
TEST_P(TinyTorchClampOps, IntegerBoundsSaturate)
{
    Tensor a = tttest::make_tensor({10, 250}, {2}, device(), tinytorch::kUInt8);
    EXPECT_EQ(tttest::to_double_vec(tinytorch::clamp(a, -5, 300)), (std::vector<double>{10, 250}));
    EXPECT_EQ(tttest::to_double_vec(tinytorch::clamp(a, 20, 100)), (std::vector<double>{20, 100}));
    Tensor b = tttest::make_tensor({1, 2, 3}, {3}, device(), tinytorch::kInt32);
    EXPECT_EQ(tttest::to_double_vec(tinytorch::clamp(b, 1.5, 2.5)), (std::vector<double>{2, 2, 2}));
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPoissonNoise);
TEST_P(TinyTorchPoissonNoise, AllFloatingTypes)
{
    for (auto dtype : {tinytorch::kHalf, tinytorch::kFloat, tinytorch::kDouble})
    {
        Tensor a = tinytorch::full({20000}, 100, TensorOptions().device(device()).dtype(dtype));
        tinytorch::add_poisson_noise_(a);
        Tensor f = a.to(tinytorch::kDouble);
        EXPECT_NEAR(f.mean().toDouble(), 100, 1) << dtype;
        EXPECT_NEAR(f.std().toDouble(), 10, 1) << dtype;
    }
}

// ------------------------------------------------------------------ determinism
// Global reductions and the autograd traversal must be bit-reproducible across
// repeated runs at a fixed configuration (thread count / grid geometry).

namespace
{
// Fixed non-trivial input (varied signs/magnitudes) so FP accumulation order is observable.
std::vector<float> det_input(int n)
{
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i)
    {
        v[i] = float((i % 97) - 48) * (1.0f + 0.125f * float(i % 7));
    }
    return v;
}
}  // namespace

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchDeterminism);

TEST_P(TinyTorchDeterminism, GlobalSum)
{
    int n      = 10000;
    Tensor a   = tttest::make_tensor(det_input(n), SizeType{n}, device());
    Tensor ref = tinytorch::sum(a);
    for (int k = 1; k < 16; ++k)
    {
        TT_EXPECT_CLOSE(tinytorch::sum(a), ref, 0.0, 0.0);
    }
}

TEST_P(TinyTorchDeterminism, AbsSum)
{
    int n      = 10000;
    Tensor a   = tttest::make_tensor(det_input(n), SizeType{n}, device());
    Tensor ref = tinytorch::abs_sum(a);
    for (int k = 1; k < 16; ++k)
    {
        TT_EXPECT_CLOSE(tinytorch::abs_sum(a), ref, 0.0, 0.0);
    }
}

TEST_P(TinyTorchDeterminism, ProdSum)
{
    int n      = 10000;
    Tensor a   = tttest::make_tensor(det_input(n), SizeType{n}, device());
    Tensor ref = tinytorch::prod_sum(a);
    for (int k = 1; k < 16; ++k)
    {
        TT_EXPECT_CLOSE(tinytorch::prod_sum(a), ref, 0.0, 0.0);
    }
}

TEST_P(TinyTorchDeterminism, Mean)
{
    int n      = 10000;
    Tensor a   = tttest::make_tensor(det_input(n), SizeType{n}, device());
    Tensor ref = tinytorch::mean(a);
    for (int k = 1; k < 16; ++k)
    {
        TT_EXPECT_CLOSE(tinytorch::mean(a), ref, 0.0, 0.0);
    }
}

TEST_P(TinyTorchDeterminism, Std)
{
    int n      = 10000;
    Tensor a   = tttest::make_tensor(det_input(n), SizeType{n}, device());
    Tensor ref = tinytorch::std(a);
    for (int k = 1; k < 16; ++k)
    {
        TT_EXPECT_CLOSE(tinytorch::std(a), ref, 0.0, 0.0);
    }
}

TEST_P(TinyTorchDeterminism, Backward)
{
    // Identical inputs must give bit-identical leaf gradients across independent
    // forward+backward cycles (guards the sequence_nr / grad_map ordering).
    auto run = [&]() -> std::vector<std::vector<double>>
    {
        Tensor x    = tttest::leaf({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, device());
        Tensor y    = tttest::leaf({0.5f, 0.1f, 0.9f, 0.3f, 0.7f, 0.2f}, {2, 3}, device());
        Tensor loss = tinytorch::sum(x * x + x * y + y);  // x used twice -> accumulation
        tinytorch::backward(loss);

        std::vector<std::vector<double>> out;
        out.push_back(tttest::to_double_vec(x.grad()));
        out.push_back(tttest::to_double_vec(y.grad()));
        return out;
    };

    std::vector<std::vector<double>> r1 = run();
    std::vector<std::vector<double>> r2 = run();
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t p = 0; p < r1.size(); ++p)
    {
        ASSERT_EQ(r1[p].size(), r2[p].size());
        for (size_t i = 0; i < r1[p].size(); ++i)
        {
            EXPECT_EQ(r1[p][i], r2[p][i]) << "param " << p << " grad[" << i << "]";
        }
    }
}

TEST_P(TinyTorchMinMaxReduce, DoubleValuesAndIndices)
{
    Tensor a = tttest::make_tensor({1.0, 2.0, -3.5, 1e30, -1e-30, 7.0}, {2, 3}, device(), tinytorch::kDouble);

    auto [mn, mn_idx] = tinytorch::min(a, 1, false);
    auto [mx, mx_idx] = tinytorch::max(a, 1, false);
    EXPECT_EQ(tttest::to_double_vec(mn), (std::vector<double>{-3.5, double(-1e-30f)}));
    EXPECT_EQ(tttest::to_double_vec(mx), (std::vector<double>{2.0, double(1e30f)}));
    EXPECT_EQ(tttest::to_double_vec(mn_idx), (std::vector<double>{2, 1}));
    EXPECT_EQ(tttest::to_double_vec(mx_idx), (std::vector<double>{1, 0}));
}

// ties select the first index (as on the CPU and in PyTorch)
TEST_P(TinyTorchMinMaxReduce, TiesSelectFirstIndex)
{
    for (auto dtype : {tinytorch::kFloat, tinytorch::kDouble})
    {
        Tensor a          = tttest::make_tensor({5, 1, 5, 1}, {1, 4}, device(), dtype);
        auto [mn, mn_idx] = tinytorch::min(a, 1, false);
        auto [mx, mx_idx] = tinytorch::max(a, 1, false);
        EXPECT_EQ(tttest::to_double_vec(mn_idx), (std::vector<double>{1})) << dtype;
        EXPECT_EQ(tttest::to_double_vec(mx_idx), (std::vector<double>{0})) << dtype;
    }
}

TEST_P(TinyTorchStdOps, Half)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device(), tinytorch::kHalf);
    // population std, as for float
    EXPECT_NEAR(a.std().to(kFloat).toFloat(), std::sqrt(5.0 / 4.0), 1e-2);
}

// an infinite bound means unbounded, also for half (std::numeric_limits is not specialized for it)
TEST_P(TinyTorchTensorClampMinMethods, InfiniteBound)
{
    for (auto dtype : {tinytorch::kHalf, tinytorch::kFloat})
    {
        Tensor a = tttest::make_tensor({-2, -0.5, 0.5, 2}, {4}, device(), dtype);
        TT_EXPECT_CLOSE(a.clamp_min(0.25).to(kFloat), tttest::make_tensor({0.25, 0.25, 0.5, 2}, {4}, device()), 0, 0);
        TT_EXPECT_CLOSE(a.clamp_max(0.25).to(kFloat), tttest::make_tensor({-2, -0.5, 0.25, 0.25}, {4}, device()), 0, 0);
    }
}
