/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

// ------------------------------------------------------------------ Adam

TEST(TinyTorchAdam, ConvergesToKnownMinimum)
{
    // minimize (w - 3)^2  ->  w* = 3
    Tensor w = tttest::leaf({0.0f}, {1}, kCPU);

    tinytorch::optim::Adam optim({w}, tinytorch::optim::AdamOptions(0.1));

    for (int i = 0; i < 200; ++i)
    {
        optim.zero_grad();
        Tensor loss = (w - 3.0) * (w - 3.0);
        tinytorch::backward(loss);
        optim.step();
    }

    EXPECT_NEAR(w.toFloat(), 3.0, 0.05);
}

TEST(TinyTorchAdam, ZeroGradClearsGradient)
{
    Tensor w = tttest::leaf({0.0f}, {1}, kCPU);
    tinytorch::optim::Adam optim({w}, tinytorch::optim::AdamOptions(0.1));

    Tensor loss = (w - 3.0) * (w - 3.0);
    tinytorch::backward(loss);
    EXPECT_FLOAT_EQ(w.grad().toFloat(), -6.0f);  // d(w-3)^2/dw = 2(w-3) at w=0

    optim.zero_grad();
    EXPECT_FLOAT_EQ(w.grad().toFloat(), 0.0f);
}

TEST(TinyTorchAdam, OptionsDefaults)
{
    tinytorch::optim::AdamOptions o;
    EXPECT_DOUBLE_EQ(o.lr(), 1e-3);
    EXPECT_DOUBLE_EQ(std::get<0>(o.betas()), 0.9);
    EXPECT_DOUBLE_EQ(std::get<1>(o.betas()), 0.999);
    EXPECT_DOUBLE_EQ(o.eps(), 1e-8);
    EXPECT_DOUBLE_EQ(o.weight_decay(), 0.0);
}

// ------------------------------------------------------------------ SGD

// Hand-verified single step:
//   w0 = 0, g0 = -6  (loss (w-3)^2)
//   step 0: b = g = -6;  nesterov: g = g + 0.9*b = -11.4;  w1 = 0 - 0.1*(-11.4) = 1.14
TEST(TinyTorchSGD, SingleStepHandVerified)
{
    Tensor w = tttest::leaf({0.0f}, {1}, kCPU);

    tinytorch::optim::SGDOptimizer optim({w}, 0.1f);
    // defaults: momentum = 0.9, dampening = 0.1, nesterov = true

    Tensor loss = (w - 3.0) * (w - 3.0);
    tinytorch::backward(loss);
    optim.Step();

    EXPECT_NEAR(w.toFloat(), 1.14f, 1e-5f);
}

TEST(TinyTorchSGD, ConvergesToKnownMinimum)
{
    Tensor w = tttest::leaf({0.0f}, {1}, kCPU);
    tinytorch::optim::SGDOptimizer optim({w}, 0.1f);

    for (int i = 0; i < 300; ++i)
    {
        optim.zero_grad();
        Tensor loss = (w - 3.0) * (w - 3.0);
        tinytorch::backward(loss);
        optim.Step();
    }

    EXPECT_NEAR(w.toFloat(), 3.0, 0.2);
}

// zero_grad() must clear the gradients of the optimizer's parameters
TEST(TinyTorchSGD, ZeroGradIsLowercase)
{
    Tensor w = tttest::leaf({0.0f}, {1}, kCPU);
    tinytorch::optim::SGDOptimizer optim({w}, 0.1f);

    Tensor loss = (w - 3.0) * (w - 3.0);
    tinytorch::backward(loss);
    // grad of (w-3)^2 at w=0 is -6
    EXPECT_NEAR(tttest::to_double_vec(w.grad())[0], -6.0, 1e-5);

    optim.zero_grad();
    TT_EXPECT_CLOSE(w.grad(), tinytorch::zeros({1}, TensorOptions().device(kCPU)), 0, 0);
}

// ------------------------------------------------------------------ multi param

TEST(TinyTorchOptimizer, MultipleParams)
{
    // minimize (w0-1)^2 + (w1+2)^2  ->  w0*=1, w1*=-2
    Tensor w0 = tttest::leaf({0.0f}, {1}, kCPU);
    Tensor w1 = tttest::leaf({0.0f}, {1}, kCPU);

    tinytorch::optim::Adam optim({w0, w1}, tinytorch::optim::AdamOptions(0.1));

    for (int i = 0; i < 200; ++i)
    {
        optim.zero_grad();
        Tensor loss = (w0 - 1.0) * (w0 - 1.0) + (w1 + 2.0) * (w1 + 2.0);
        tinytorch::backward(loss);
        optim.step();
    }

    EXPECT_NEAR(w0.toFloat(), 1.0, 0.05);
    EXPECT_NEAR(w1.toFloat(), -2.0, 0.05);
}
