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

// d(sqrt(a))/da = 1 / (2 * sqrt(a));  a = 4  ->  0.25
TEST(Backward, Sqrt)
{
    Tensor a = tttest::leaf({4.0f}, {1}, kCPU);

    tinytorch::backward(tinytorch::sqrt(a));

    EXPECT_FLOAT_EQ(a.grad().toFloat(), 0.25f);
}

// d(exp(a))/da = exp(a)
TEST(Backward, Exp)
{
    Tensor a = tttest::leaf({1.0f}, {1}, kCPU);

    tinytorch::backward(tinytorch::exp(a));

    EXPECT_NEAR(a.grad().toFloat(), std::exp(1.0f), 1e-6f);
}

// d(log(a))/da = 1 / a
TEST(Backward, Log)
{
    Tensor a = tttest::leaf({2.0f}, {1}, kCPU);

    tinytorch::backward(tinytorch::log(a));

    EXPECT_FLOAT_EQ(a.grad().toFloat(), 0.5f);
}

// d(relu(a))/da = 1 if a > 0 else 0
TEST(Backward, Relu)
{
    Tensor a = tttest::leaf({2.0f}, {1}, kCPU);
    Tensor b = tttest::leaf({-2.0f}, {1}, kCPU);

    tinytorch::backward(tinytorch::relu(a) + tinytorch::relu(b));

    EXPECT_FLOAT_EQ(a.grad().toFloat(), 1.0f);
    EXPECT_FLOAT_EQ(b.grad().toFloat(), 0.0f);
}

}  // namespace

// ------------------------------------------------------------------ elementwise

TT_INSTANTIATE_DEVICE_TESTS(UnaryAbs);
TEST_P(UnaryAbs, ValuesAndGradient)
{
    Tensor a = tttest::leaf({-2, 1, -3, 4}, {2, 2}, device());

    Tensor r = tinytorch::abs(a);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({2, 1, 3, 4}, {2, 2}, device()), 0, 0);

    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({-1, 1, -1, 1}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(UnarySqrt);
TEST_P(UnarySqrt, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 4, 9, 16}, {2, 2}, device());

    Tensor r = tinytorch::sqrt(a);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device()), 1e-5, 1e-5);

    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({0.5, 0.25, 1.0 / 6, 0.125}, {2, 2}, device()), 1e-4, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(UnarySquare);
TEST_P(UnarySquare, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, -2, 3, -4}, {2, 2}, device());

    Tensor r = a.square();
    TT_EXPECT_CLOSE(r, tttest::make_tensor({1, 4, 9, 16}, {2, 2}, device()), 0, 0);

    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({2, -4, 6, -8}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(UnaryLog);
TEST_P(UnaryLog, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 2, 4, 8}, {2, 2}, device());

    Tensor r = tinytorch::log(a);
    std::vector<float> expected = {0.0f, std::log(2.0f), std::log(4.0f), std::log(8.0f)};
    TT_EXPECT_CLOSE(r, tttest::make_tensor(expected, {2, 2}, device()), 1e-5, 1e-6);

    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({1, 0.5, 0.25, 0.125}, {2, 2}, device()), 1e-4, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(UnaryLog1p);
TEST_P(UnaryLog1p, Values)
{
    Tensor a = tttest::make_tensor({0, 1, 2, 3}, {2, 2}, device());
    TT_EXPECT_CLOSE(tinytorch::log1p(a),
                    tttest::make_tensor({0, std::log(2.0), std::log(3.0), std::log(4.0)}, {2, 2}, device()), 1e-5,
                    1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(UnaryExp);
TEST_P(UnaryExp, ValuesAndGradient)
{
    Tensor a = tttest::leaf({0, 1, 2, 3}, {2, 2}, device());

    Tensor r = tinytorch::exp(a);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({1, std::exp(1.0), std::exp(2.0), std::exp(3.0)}, {2, 2}, device()),
                    1e-5, 1e-6);

    tinytorch::backward(tinytorch::sum(r));
    // d(exp(a))/da = exp(a)
    TT_EXPECT_CLOSE(a.grad(), r, 1e-4, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(UnarySinCos);
TEST_P(UnarySinCos, ValuesAndGradient)
{
    Tensor a = tttest::leaf({0, 1, 2, 3}, {2, 2}, device());

    Tensor s = tinytorch::sin(a);
    TT_EXPECT_CLOSE(s, tttest::make_tensor({0, std::sin(1.0), std::sin(2.0), std::sin(3.0)}, {2, 2}, device()),
                    1e-5, 1e-6);

    Tensor c = tinytorch::cos(a);
    TT_EXPECT_CLOSE(c, tttest::make_tensor({1, std::cos(1.0), std::cos(2.0), std::cos(3.0)}, {2, 2}, device()),
                    1e-5, 1e-6);

    // d(sin)/dx = cos,  d(cos)/dx = -sin
    a.mutable_grad().zero_();
    tinytorch::backward(tinytorch::sum(s));
    TT_EXPECT_CLOSE(a.grad(), c, 1e-4, 1e-6);

    a.mutable_grad().zero_();
    tinytorch::backward(tinytorch::sum(c));
    TT_EXPECT_CLOSE(a.grad(), -s, 1e-4, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(UnarySignRound);
TEST_P(UnarySignRound, Values)
{
    Tensor a = tttest::make_tensor({-1.5, 0, 2.7, -0.2}, {2, 2}, device());

    TT_EXPECT_CLOSE(tinytorch::sign(a), tttest::make_tensor({-1, 0, 1, -1}, {2, 2}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::round(a), tttest::make_tensor({-2, 0, 3, 0}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(UnaryRelu);
TEST_P(UnaryRelu, ValuesAndGradient)
{
    Tensor a = tttest::leaf({-1, 0.5, 2, -0.5}, {2, 2}, device());

    Tensor r = tinytorch::relu(a);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({0, 0.5, 2, 0}, {2, 2}, device()), 0, 0);

    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({0, 1, 1, 0}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(UnarySigmoid);
TEST_P(UnarySigmoid, ValuesAndGradient)
{
    Tensor a = tttest::leaf({-1, 0, 1, 2}, {2, 2}, device());

    Tensor r = tinytorch::sigmoid(a);
    std::vector<float> expected = {1.0f / (1.0f + std::exp(1.0f)), 0.5f, 1.0f - 1.0f / (1.0f + std::exp(1.0f)),
                                   1.0f - 1.0f / (1.0f + std::exp(2.0f))};
    TT_EXPECT_CLOSE(r, tttest::make_tensor(expected, {2, 2}, device()), 1e-5, 1e-6);

    // d(sigma(x))/dx = sigma(x) * (1 - sigma(x))
    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), r * (1.0 - r), 1e-4, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(UnarySoftplus);
TEST_P(UnarySoftplus, ValuesAndGradient)
{
    Tensor a = tttest::leaf({-1, 0, 1, 2}, {2, 2}, device());
    const double beta = 1.0;

    Tensor r = tinytorch::softplus(a, beta);
    std::vector<float> expected = {std::log1p(std::exp(-1.0f)), std::log(2.0f), std::log1p(std::exp(1.0f)),
                                   std::log1p(std::exp(2.0f))};
    TT_EXPECT_CLOSE(r, tttest::make_tensor(expected, {2, 2}, device()), 1e-5, 1e-6);

    // d(log(1+e^x))/dx = 1 / (1 + e^-x) = sigmoid(x)
    tinytorch::backward(tinytorch::sum(r));
    TT_EXPECT_CLOSE(a.grad(), tinytorch::sigmoid(a), 1e-4, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(UnarySoftmax);
TEST_P(UnarySoftmax, SumsToOneAndGradient)
{
    Tensor a = tttest::leaf({1, 2, 3, 4}, {2, 2}, device());

    Tensor r = tinytorch::softmax(a, 1);

    // each row sums to 1
    Tensor row_sums = r.sum(1, true);
    TT_EXPECT_CLOSE(row_sums, tttest::make_tensor({1, 1}, {2, 1}, device()), 1e-5, 1e-6);

    // monotone in the input along the reduced dim
    auto v = tttest::to_double_vec(r);
    EXPECT_LT(v[0], v[1]);
    EXPECT_LT(v[2], v[3]);

    // gradient matches finite differences
    tttest::check_grads({a},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::softmax(p[0], 1));
                        },
                        2e-2, 2e-3);
}
