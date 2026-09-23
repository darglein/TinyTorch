/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

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
    TT_EXPECT_CLOSE(s0, tttest::make_tensor({5, 7, 9}, {1,  3}, device()), 1e-5, 1e-6);

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

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPowTensor);
TEST_P(TinyTorchPowTensor, ValuesAndGradient)
{
    Tensor a = tttest::leaf({2, 3}, {2}, device());
    Tensor b = tttest::leaf({2, 3}, {2}, device());

    Tensor r = tinytorch::pow(a, b);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({4, 27}, {2}, device()), 1e-5, 1e-6);

    tttest::check_grads({a, b},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::pow(p[0], p[1]));
                        },
                        2e-2, 2e-3);
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
    tttest::check_grads({a},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::abs_sum(p[0]);
                        },
                        2e-2, 2e-3);
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
    tttest::check_grads({a},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::prod_sum(p[0]);
                        },
                        2e-2, 2e-3);
}

// ------------------------------------------------------------------ median

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMedian);
TEST_P(TinyTorchMedian, Values)
{
    Tensor a = tttest::make_tensor({3, 1, 2}, {3}, device());
    TT_EXPECT_CLOSE(tinytorch::median(a, 0.5), tttest::make_tensor({2}, {1}, device()), 1e-5, 1e-6);
}
