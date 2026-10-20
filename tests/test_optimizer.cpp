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
    EXPECT_FALSE(o.amsgrad());
}

namespace
{
// Reference of TinyTorch's Adam update for a single weight and a fixed gradient sequence (the bias correction is
// folded into the learning rate, eps is added to sqrt(m2 + 1e-10)). The betas are used as float, like in Adam::step.
double AdamReference(double w, const std::vector<double>& grads, double lr, double weight_decay, bool amsgrad)
{
    double b1 = float(0.9), b2 = float(0.999), eps = 1e-8;
    double m1 = 0, m2 = 0, m2_max = 0;
    for (int t = 1; t <= (int)grads.size(); ++t)
    {
        double g = grads[t - 1] + weight_decay * w;
        m1       = b1 * m1 + (1 - b1) * g;
        m2       = b2 * m2 + (1 - b2) * g * g;
        m2_max   = std::max(m2_max, m2);
        double v = amsgrad ? m2_max : m2;
        double l = lr * std::sqrt(1.0 - std::pow(b2, double(t))) / (1.0 - std::pow(b1, double(t)));
        w -= std::min(std::max(l / (std::sqrt(v + 1e-10) + eps), 0.0), 1e36) * m1;
    }
    return w;
}
}  // namespace

// weight_decay (L2, added to the gradient) and amsgrad (max of the second moments) follow the PyTorch semantics
TT_INSTANTIATE_DEVICE_TESTS(TinyTorchAdamOptionsUsed);
TEST_P(TinyTorchAdamOptionsUsed, WeightDecayAndAmsgrad)
{
    std::vector<double> grads(30, 0.01);
    grads[0] = 1.0;
    grads[3] = -0.5;
    EXPECT_GT(std::abs(AdamReference(1, grads, 0.1, 0, true) - AdamReference(1, grads, 0.1, 0, false)), 1e-3);
    EXPECT_GT(std::abs(AdamReference(1, grads, 0.1, 0.5, false) - AdamReference(1, grads, 0.1, 0, false)), 1e-3);

    for (double weight_decay : {0.0, 0.5})
    {
        for (bool amsgrad : {false, true})
        {
            Tensor w = tttest::leaf({1.0f}, {1}, device(), tinytorch::kDouble);
            tinytorch::optim::AdamOptions options(0.1);
            options.weight_decay(weight_decay);
            options.amsgrad(amsgrad);
            tinytorch::optim::Adam optim({w}, options);
            for (double g : grads)
            {
                optim.zero_grad();
                tinytorch::backward(tinytorch::sum(w * g));
                optim.step();
            }
            EXPECT_NEAR(w.toDouble(), AdamReference(1, grads, 0.1, weight_decay, amsgrad), 1e-6)
                << "weight_decay " << weight_decay << " amsgrad " << amsgrad;
        }
    }
}

// Parameters without a gradient (unused or zero_grad(set_to_none)) are skipped
TEST(TinyTorchAdam, SkipsParamsWithoutGrad)
{
    Tensor w0 = tttest::leaf({1.0f}, {1}, kCPU);
    Tensor w1 = tttest::make_tensor({2.0f}, {1}, kCPU);
    w1.set_requires_grad(true, false);
    ASSERT_FALSE(w1.grad().defined());

    tinytorch::optim::Adam adam({w0, w1}, tinytorch::optim::AdamOptions(0.1));
    tinytorch::optim::SGDOptimizer sgd({w0, w1}, 0.1f);
    tinytorch::backward(tinytorch::sum(w0 * 2.0));
    adam.step();
    sgd.Step();
    EXPECT_LT(w0.toFloat(), 1.0f);
    EXPECT_FLOAT_EQ(w1.toFloat(), 2.0f);
}

// The optimizer state follows the parameters when they are moved after the optimizer was created (e.g. module.to())
TEST(TinyTorchAdam, StateFollowsParamDeviceAndDtype)
{
    std::vector<double> grads   = {1.0, 0.3, -0.5, 0.2};
    std::vector<Device> devices = {kCPU};
    if (tttest::has_cuda()) devices.push_back(kCUDA);
    for (Device device : devices)
    {
        Tensor w = tttest::leaf({1.0f}, {1}, kCPU, kFloat);
        tinytorch::optim::Adam optim({w}, tinytorch::optim::AdamOptions(0.1));
        for (int i = 0; i < (int)grads.size(); ++i)
        {
            if (i == 2)
            {
                w.to_(tinytorch::kDouble);
                w.to_(device);
            }
            optim.zero_grad();
            tinytorch::backward(tinytorch::sum(w * grads[i]));
            optim.step();
            EXPECT_EQ(optim.m1[0].device(), w.device());
            EXPECT_EQ(optim.m2[0].dtype(), w.dtype());
        }
        EXPECT_EQ(w.device(), device);
        EXPECT_NEAR(w.toDouble(), AdamReference(1, grads, 0.1, 0, false), 1e-5);
    }
}

// The bias correction uses the step count of each parameter (like PyTorch): a parameter added with add_param_group
// after some steps, or one that had no gradient in the previous steps, gets the bias correction of its first step
TT_INSTANTIATE_DEVICE_TESTS(TinyTorchAdamStepCount);
TEST_P(TinyTorchAdamStepCount, BiasCorrectionPerParameter)
{
    tinytorch::optim::AdamOptions options(0.1);
    double first_step = AdamReference(1, {1.0}, 0.1, 0, false);

    Tensor a = tttest::leaf({1.0f}, {1}, device());
    tinytorch::optim::Adam adam({a}, options);
    for (int i = 0; i < 20; ++i)
    {
        a.mutable_grad().fill_(1);
        adam.step();
    }
    Tensor b = tttest::leaf({1.0f}, {1}, device());
    adam.add_param_group(
        std::make_pair(std::vector<Tensor>{b}, std::make_unique<tinytorch::optim::AdamOptions>(options)));
    a.mutable_grad().fill_(1);
    b.mutable_grad().fill_(1);
    adam.step();
    EXPECT_NEAR(b.toDouble(), first_step, 1e-5);

    Tensor c = tttest::leaf({1.0f}, {1}, device());
    Tensor d = tttest::leaf({1.0f}, {1}, device());
    tinytorch::optim::Adam adam2({c, d}, options);
    for (int i = 0; i < 20; ++i)
    {
        c.mutable_grad().fill_(1);
        if (d.grad().defined()) d.mutable_grad() = Tensor();
        adam2.step();
    }
    d.set_grad(tttest::make_tensor({1}, {1}, device()));
    adam2.step();
    EXPECT_NEAR(d.toDouble(), first_step, 1e-5);
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

// SGD works for every float dtype and device
TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSGDDevice);
TEST_P(TinyTorchSGDDevice, TwoStepsHandVerified)
{
    for (ScalarType dtype : {kFloat, tinytorch::kDouble})
    {
        Tensor w = tttest::leaf({0.0f}, {1}, device(), dtype);
        tinytorch::optim::SGDOptimizer optim({w}, 0.1f);

        tinytorch::backward((w - 3.0) * (w - 3.0));
        optim.Step();
        EXPECT_NEAR(w.toDouble(), 1.14, 1e-5);

        // step 1: g = 2 * (1.14 - 3) = -3.72;  b = 0.9 * -6 + 0.9 * -3.72 = -8.748
        //         nesterov: g = -3.72 + 0.9 * -8.748 = -11.5932;  w = 1.14 + 1.15932 = 2.29932
        optim.zero_grad();
        tinytorch::backward((w - 3.0) * (w - 3.0));
        optim.Step();
        EXPECT_NEAR(w.toDouble(), 2.29932, 1e-5);
        EXPECT_EQ(w.dtype(), dtype);
    }
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
