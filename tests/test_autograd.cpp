/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

namespace
{

// Folds in the cases from the former tests/test_backward.cpp.

// The graph is one-shot: forward must be re-run every iteration, gradients
// accumulate on the leaf unless zeroed.
TEST(TinyTorchAutograd, GradientAccumulation)
{
    Tensor a = tttest::leaf({2.0f}, {1}, kCPU);

    for (int i = 0; i < 3; ++i)
    {
        tinytorch::backward(a * a);
    }

    // d(a^2)/da = 2a = 4 per iteration, 3 iterations -> 12
    EXPECT_FLOAT_EQ(a.grad().toFloat(), 12.0f);
}

// An explicit upstream gradient is propagated: backward(loss, grad)
TEST(TinyTorchAutograd, CustomUpstreamGradient)
{
    Tensor a = tttest::leaf({2.0f}, {1}, kCPU);

    Tensor loss = a * a;
    Tensor g    = tinytorch::full({1}, 3.0f);
    tinytorch::backward(loss, g);

    // d(a^2)/da = 2a = 4, scaled by the upstream grad 3 -> 12
    EXPECT_FLOAT_EQ(a.grad().toFloat(), 12.0f);
}

}  // namespace

// ------------------------------------------------------------------ grad mode

TEST(TinyTorchAutograd, GradModeDefaults)
{
    // grad mode is on by default
    EXPECT_TRUE(tinytorch::GradMode::is_enabled());

    Tensor a = tttest::leaf({2.0f}, {1}, kCPU);
    Tensor b = a * 3.0;
    EXPECT_TRUE(b.requires_grad());
}

TEST(TinyTorchAutograd, NoGradGuardDisablesTracking)
{
    {
        tinytorch::NoGradGuard ngg;
        EXPECT_FALSE(tinytorch::GradMode::is_enabled());

        Tensor a = tttest::leaf({2.0f}, {1}, kCPU);
        Tensor b = a * 3.0;
        // under the guard no node is built and no edge is set
        EXPECT_FALSE(b.requires_grad());
        EXPECT_EQ(b.getEdge(), nullptr);
    }

    // after the guard the mode is restored
    EXPECT_TRUE(tinytorch::GradMode::is_enabled());
}

TEST(TinyTorchAutograd, AutoGradModeRestoresPreviousState)
{
    {
        tinytorch::AutoGradMode on(true);
        EXPECT_TRUE(tinytorch::GradMode::is_enabled());

        tinytorch::AutoGradMode off(false);
        EXPECT_FALSE(tinytorch::GradMode::is_enabled());
    }
    EXPECT_TRUE(tinytorch::GradMode::is_enabled());
}

// ------------------------------------------------------------------ leaf properties

TEST(TinyTorchAutograd, LeafProperties)
{
    Tensor t = tttest::leaf({1, 2}, {2}, kCPU);

    EXPECT_TRUE(t.requires_grad());
    EXPECT_TRUE(t.is_leaf());
    EXPECT_TRUE(t.grad().defined());

    // zero-initialized gradient
    TT_EXPECT_CLOSE(t.grad(), tinytorch::zeros({2}, TensorOptions().device(kCPU)), 0, 0);
}

TEST(TinyTorchAutograd, BackwardRequiresEdge)
{
    // A tensor that never required grad has no edge; using it as a loss must fail the
    // CHECK in backward(). We only verify the preconditions here (the CHECK itself
    // aborts and cannot be tested in-process).
    Tensor t = tinytorch::zeros({1});
    EXPECT_FALSE(t.requires_grad());
    EXPECT_EQ(t.getEdge(), nullptr);

    Tensor t2 = tttest::leaf({1.0f}, {1}, kCPU);
    Tensor loss = t2 * 2.0;
    EXPECT_TRUE(loss.requires_grad());
    EXPECT_TRUE(loss.getEdge());
}

// ------------------------------------------------------------------ retain_grad flag

TEST(TinyTorchAutograd, BackwardWithExplicitGradOnVectorLoss)
{
    // backward(loss, grad) with a non-scalar loss requires matching shapes
    Tensor a = tttest::leaf({1, 2}, {2}, kCPU);
    Tensor loss = a * a;  // elementwise, shape {2}
    Tensor g = tinytorch::full({2}, 1.0f);

    tinytorch::backward(loss, g);
    // d(a^2)/da = 2a
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({2, 4}, {2}, kCPU), 0, 0);
}
