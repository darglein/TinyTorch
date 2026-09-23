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

// d(a+b)/da = 1, d(a+b)/db = 1
TEST(TinyTorchBackward, Add)
{
    Tensor a = tttest::leaf({3.0f}, {1}, kCPU);
    Tensor b = tttest::leaf({2.0f}, {1}, kCPU);

    Tensor loss = a + b;
    tinytorch::backward(loss);

    EXPECT_FLOAT_EQ(a.grad().toFloat(), 1.0f);
    EXPECT_FLOAT_EQ(b.grad().toFloat(), 1.0f);
}

// d(a-b)/da = 1, d(a-b)/db = -1
TEST(TinyTorchBackward, Sub)
{
    Tensor a = tttest::leaf({3.0f}, {1}, kCPU);
    Tensor b = tttest::leaf({2.0f}, {1}, kCPU);

    tinytorch::backward(a - b);

    EXPECT_FLOAT_EQ(a.grad().toFloat(), 1.0f);
    EXPECT_FLOAT_EQ(b.grad().toFloat(), -1.0f);
}

// d(a*b)/da = b, d(a*b)/db = a
TEST(TinyTorchBackward, Mult)
{
    Tensor a = tttest::leaf({3.0f}, {1}, kCPU);
    Tensor b = tttest::leaf({4.0f}, {1}, kCPU);

    tinytorch::backward(a * b);

    EXPECT_FLOAT_EQ(a.grad().toFloat(), 4.0f);
    EXPECT_FLOAT_EQ(b.grad().toFloat(), 3.0f);
}

// d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
TEST(TinyTorchBackward, Div)
{
    Tensor a = tttest::leaf({6.0f}, {1}, kCPU);
    Tensor b = tttest::leaf({3.0f}, {1}, kCPU);

    tinytorch::backward(a / b);

    EXPECT_FLOAT_EQ(a.grad().toFloat(), 1.0f / 3.0f);
    EXPECT_FLOAT_EQ(b.grad().toFloat(), -2.0f / 3.0f);
}

}  // namespace

// ------------------------------------------------------------------ binary ops

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchBinaryOpAdd);
TEST_P(TinyTorchBinaryOpAdd, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 2, 3, 4}, {2, 2}, device());
    Tensor b = tttest::leaf({10, 20, 30, 40}, {2, 2}, device());

    Tensor sum = a + b;
    TT_EXPECT_CLOSE(sum, tttest::make_tensor({11, 22, 33, 44}, {2, 2}, device()), 1e-6, 1e-5);

    tinytorch::backward(tinytorch::sum(sum));
    TT_EXPECT_CLOSE(a.grad(), tinytorch::ones({2, 2}, TensorOptions().device(device())), 0, 0);
    TT_EXPECT_CLOSE(b.grad(), tinytorch::ones({2, 2}, TensorOptions().device(device())), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchBinaryOpSub);
TEST_P(TinyTorchBinaryOpSub, ValuesAndGradient)
{
    Tensor a = tttest::leaf({10, 20, 30, 40}, {2, 2}, device());
    Tensor b = tttest::leaf({1, 2, 3, 4}, {2, 2}, device());

    Tensor diff = a - b;
    TT_EXPECT_CLOSE(diff, tttest::make_tensor({9, 18, 27, 36}, {2, 2}, device()), 1e-6, 1e-5);

    tinytorch::backward(tinytorch::sum(diff));
    TT_EXPECT_CLOSE(a.grad(), tinytorch::ones({2, 2}, TensorOptions().device(device())), 0, 0);
    TT_EXPECT_CLOSE(b.grad(), -tinytorch::ones({2, 2}, TensorOptions().device(device())), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchBinaryOpMult);
TEST_P(TinyTorchBinaryOpMult, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 2, 3, 4}, {2, 2}, device());
    Tensor b = tttest::leaf({10, 20, 30, 40}, {2, 2}, device());

    Tensor prod = a * b;
    TT_EXPECT_CLOSE(prod, tttest::make_tensor({10, 40, 90, 160}, {2, 2}, device()), 1e-6, 1e-5);

    tinytorch::backward(tinytorch::sum(prod));
    TT_EXPECT_CLOSE(a.grad(), b, 0, 0);
    TT_EXPECT_CLOSE(b.grad(), a, 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchBinaryOpDiv);
TEST_P(TinyTorchBinaryOpDiv, ValuesAndGradient)
{
    Tensor a = tttest::leaf({10, 20, 30, 40}, {2, 2}, device());
    Tensor b = tttest::leaf({1, 2, 3, 4}, {2, 2}, device());

    Tensor quot = a / b;
    TT_EXPECT_CLOSE(quot, tttest::make_tensor({10, 10, 10, 10}, {2, 2}, device()), 1e-6, 1e-5);

    tinytorch::backward(tinytorch::sum(quot));
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({1, 0.5, 1.0 / 3, 0.25}, {2, 2}, device()), 1e-4, 1e-6);
    TT_EXPECT_CLOSE(b.grad(), tttest::make_tensor({-10, -5, -10.0 / 3, -2.5}, {2, 2}, device()), 1e-4, 1e-6);
}

// ------------------------------------------------------------------ scalar ops

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchTensorScalarOps);
TEST_P(TinyTorchTensorScalarOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    TT_EXPECT_CLOSE(a + 1.0, tttest::make_tensor({2, 3, 4, 5}, {2, 2}, device()), 1e-6, 1e-5);
    TT_EXPECT_CLOSE(1.0 + a, tttest::make_tensor({2, 3, 4, 5}, {2, 2}, device()), 1e-6, 1e-5);
    TT_EXPECT_CLOSE(a - 1.0, tttest::make_tensor({0, 1, 2, 3}, {2, 2}, device()), 1e-6, 1e-5);
    TT_EXPECT_CLOSE(1.0 - a, tttest::make_tensor({0, -1, -2, -3}, {2, 2}, device()), 1e-6, 1e-5);
    TT_EXPECT_CLOSE(a * 2.0, tttest::make_tensor({2, 4, 6, 8}, {2, 2}, device()), 1e-6, 1e-5);
    TT_EXPECT_CLOSE(2.0 * a, tttest::make_tensor({2, 4, 6, 8}, {2, 2}, device()), 1e-6, 1e-5);
    TT_EXPECT_CLOSE(a / 2.0, tttest::make_tensor({0.5, 1, 1.5, 2}, {2, 2}, device()), 1e-6, 1e-5);
    TT_EXPECT_CLOSE(4.0 / a, tttest::make_tensor({4, 2, 4.0 / 3, 1}, {2, 2}, device()), 1e-5, 1e-5);
    TT_EXPECT_CLOSE(-a, tttest::make_tensor({-1, -2, -3, -4}, {2, 2}, device()), 1e-6, 1e-5);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchTensorScalarGradient);
TEST_P(TinyTorchTensorScalarGradient, MultAndDiv)
{
    // d(a * c)/da = c
    Tensor a = tttest::leaf({1, 2}, {2}, device());
    tinytorch::backward(tinytorch::sum(a * 3.0));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({3, 3}, {2}, device()), 0, 0);

    // d(c / a)/da = -c / a^2
    Tensor b = tttest::leaf({2, 4}, {2}, device());
    tinytorch::backward(tinytorch::sum(8.0 / b));
    TT_EXPECT_CLOSE(b.grad(), tttest::make_tensor({-8.0 / 4, -8.0 / 16}, {2}, device()), 1e-4, 1e-6);
}

// ------------------------------------------------------------------ comparisons

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchComparisonOps);
TEST_P(TinyTorchComparisonOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    Tensor b = tttest::make_tensor({2, 2, 4, 4}, {2, 2}, device());

    TT_EXPECT_CLOSE(a == 2.0, tttest::make_tensor({0, 1, 0, 0}, {2, 2}, device()), 0, 0);
    TT_EXPECT_CLOSE(a < 2.5, tttest::make_tensor({1, 1, 0, 0}, {2, 2}, device()), 0, 0);
    TT_EXPECT_CLOSE(a > 2.5, tttest::make_tensor({0, 0, 1, 1}, {2, 2}, device()), 0, 0);
    // |a - b| is zero exactly where a == b (no tensor-tensor comparison op exists)
    TT_EXPECT_CLOSE((a - b).abs(), tttest::make_tensor({1, 0, 1, 0}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ compound ops

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchCompoundAssignOps);
TEST_P(TinyTorchCompoundAssignOps, Values)
{
    // In-place semantics: the same underlying tensor is modified.
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    a += 1.0;
    TT_EXPECT_CLOSE(a, tttest::make_tensor({2, 3, 4, 5}, {2, 2}, device()), 0, 0);

    a -= 2.0;
    TT_EXPECT_CLOSE(a, tttest::make_tensor({0, 1, 2, 3}, {2, 2}, device()), 0, 0);

    a *= 2.0;
    TT_EXPECT_CLOSE(a, tttest::make_tensor({0, 2, 4, 6}, {2, 2}, device()), 0, 0);

    a /= 2.0;
    TT_EXPECT_CLOSE(a, tttest::make_tensor({0, 1, 2, 3}, {2, 2}, device()), 0, 0);

    Tensor b = tttest::make_tensor({10, 20, 30, 40}, {2, 2}, device());
    a += b;
    TT_EXPECT_CLOSE(a, tttest::make_tensor({10, 21, 32, 43}, {2, 2}, device()), 0, 0);
    a -= b;
    TT_EXPECT_CLOSE(a, tttest::make_tensor({0, 1, 2, 3}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ broadcasting

// a (2,1) * b (2,3): output (2,3)
//   dL/da = [[3],[3]]       (sum-reduced over the broadcast dim)
//   dL/db = [[2,2,2],[3,3,3]] (a broadcast to the output shape)
TT_INSTANTIATE_DEVICE_TESTS(TinyTorchBroadcasting);
TEST_P(TinyTorchBroadcasting, ValuesAndReducedGradient)
{
    Tensor a = tttest::leaf({2, 3}, {2, 1}, device());
    Tensor b = tttest::leaf({1, 1, 1, 1, 1, 1}, {2, 3}, device());

    Tensor prod = a * b;
    TT_EXPECT_CLOSE(prod, tttest::make_tensor({2, 2, 2, 3, 3, 3}, {2, 3}, device()), 0, 0);

    tinytorch::backward(tinytorch::sum(prod));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({3, 3}, {2, 1}, device()), 0, 0);
    TT_EXPECT_CLOSE(b.grad(), tttest::make_tensor({2, 2, 2, 3, 3, 3}, {2, 3}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchBroadcastingAdd);
TEST_P(TinyTorchBroadcastingAdd, Values)
{
    Tensor a = tttest::make_tensor({1, 2}, {2, 1}, device());
    Tensor b = tttest::make_tensor({10, 20, 30}, {1, 3}, device());

    Tensor s = a + b;
    TT_EXPECT_CLOSE(s, tttest::make_tensor({11, 21, 31, 12, 22, 32}, {2, 3}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchBroadcastingMismatch);
TEST_P(TinyTorchBroadcastingMismatch, SameDimMustMatch)
{
    // dims must be equal or 1; (2,1) + (2,2) is valid and broadcasts
    Tensor a = tttest::make_tensor({1, 2}, {2, 1}, device());
    Tensor b = tttest::make_tensor({1, 2, 1, 2}, {2, 2}, device());
    Tensor s = a + b;
    TT_EXPECT_CLOSE(s, tttest::make_tensor({2, 3, 3, 4}, {2, 2}, device()), 0, 0);
}
