/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <atomic>
#include <thread>

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

// Like libtorch, tensor creation keeps requires_grad from the options also under a NoGradGuard
TEST(TinyTorchAutograd, CreationUnderNoGradKeepsRequiresGrad)
{
    Tensor e, z, o;
    {
        tinytorch::NoGradGuard ngg;
        auto options = TensorOptions().requires_grad(true);
        e            = tinytorch::empty({2}, options);
        z            = tinytorch::zeros({2}, options);
        o            = tinytorch::ones({2}, options);
    }
    EXPECT_TRUE(e.requires_grad());
    EXPECT_TRUE(z.requires_grad());
    EXPECT_TRUE(o.requires_grad());
    EXPECT_TRUE(o.is_leaf());

    tinytorch::backward(tinytorch::sum(o * 3.0));
    TT_EXPECT_CLOSE(o.grad(), tttest::make_tensor({3, 3}, {2}, kCPU), 0, 0);
}

namespace
{
struct CountingCallback : public tinytorch::autograd::NodeCallback
{
    std::atomic<int> forward{0}, backward{0};
    void ForwardStart(const tinytorch::autograd::NodeInfo&) override { forward++; }
    void ForwardEnd(const tinytorch::autograd::NodeInfo&) override {}
    void BackwardStart(const tinytorch::autograd::NodeInfo&) override { backward++; }
    void BackwardEnd(const tinytorch::autograd::NodeInfo&) override {}
};
}  // namespace

// The global node callback may be set while other threads run ops
TEST(TinyTorchAutograd, NodeCallbackFromThreads)
{
    CountingCallback cb;
    tinytorch::autograd::call_back = &cb;
    Tensor a                       = tttest::leaf({2.0f}, {1}, kCPU);
    tinytorch::backward(a * a);
    EXPECT_GT(cb.forward.load(), 0);
    EXPECT_GT(cb.backward.load(), 0);

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
    {
        threads.emplace_back(
            [&stop]()
            {
                tinytorch::set_num_threads(1);
                Tensor x = tttest::leaf({1.0f, 2.0f}, {2}, kCPU);
                while (!stop)
                {
                    tinytorch::backward(tinytorch::sum(x * x));
                }
            });
    }
    for (int i = 0; i < 1000; ++i)
    {
        tinytorch::autograd::call_back = (i % 2) ? &cb : nullptr;
    }
    tinytorch::autograd::call_back = nullptr;
    stop                           = true;
    for (auto& t : threads)
    {
        t.join();
    }
    EXPECT_EQ(tinytorch::autograd::call_back.load(), nullptr);
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

TEST(TinyTorchAutograd, RetainGrad)
{
    Tensor a = tttest::leaf({1, 2}, {2}, kCPU);
    Tensor b = a * 3.0;
    b.retain_grad();
    // idempotent, and b stays a non-leaf
    b.retain_grad();
    EXPECT_FALSE(b.is_leaf());

    Tensor c = b * b;
    tinytorch::backward(tinytorch::sum(c));
    // dc/db = 2b, dc/da = 6b
    TT_EXPECT_CLOSE(b.grad(), tttest::make_tensor({6, 12}, {2}, kCPU), 0, 0);
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({18, 36}, {2}, kCPU), 0, 0);

    // retain_grad on a leaf is a no-op
    a.retain_grad();
    EXPECT_TRUE(a.is_leaf());
}

// retain_grad() after the tensor was already used by other ops (also used several times)
TEST(TinyTorchAutograd, RetainGradAfterUse)
{
    Tensor a    = tttest::leaf({1, 2}, {2}, kCPU);
    Tensor b    = a * 3.0;
    Tensor c    = b * b + b;
    Tensor loss = tinytorch::sum(c);
    b.retain_grad();
    loss.retain_grad();
    tinytorch::backward(loss);
    // dc/db = 2b + 1
    TT_EXPECT_CLOSE(b.grad(), tttest::make_tensor({7, 13}, {2}, kCPU), 0, 0);
    TT_EXPECT_CLOSE(loss.grad(), tttest::make_tensor({1}, {1}, kCPU), 0, 0);
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({21, 39}, {2}, kCPU), 0, 0);
}

// The retained grad is an own tensor: it doesn't alias the gradient the engine passes on or the explicit upstream grad
TEST(TinyTorchAutograd, RetainGradDoesNotAlias)
{
    Tensor a = tttest::leaf({1, 2}, {2}, kCPU);
    Tensor b = a * 2.0;
    Tensor c = b * 3.0;
    b.retain_grad();
    c.retain_grad();
    Tensor g = tttest::make_tensor({1, 1}, {2}, kCPU);
    tinytorch::backward(c, g);
    TT_EXPECT_CLOSE(b.grad(), tttest::make_tensor({3, 3}, {2}, kCPU), 0, 0);
    g.fill_(5);
    b.mutable_grad().fill_(7);
    TT_EXPECT_CLOSE(c.grad(), tttest::make_tensor({1, 1}, {2}, kCPU), 0, 0);
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({6, 6}, {2}, kCPU), 0, 0);
}

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

// AddNode returns the same gradient tensor for both inputs: accumulating more gradient for one input must not change
// the gradient of the other one, and leaves must not share their gradient tensor
TEST(TinyTorchAutograd, SharedGradientIsNotAccumulatedInPlace)
{
    for (bool allocate_grad : {true, false})
    {
        Tensor a = tttest::make_tensor({1.0f}, {1}, kCPU);
        Tensor b = tttest::make_tensor({1.0f}, {1}, kCPU);
        a.set_requires_grad(true, allocate_grad);
        b.set_requires_grad(true, allocate_grad);

        Tensor c = a + b;
        tinytorch::backward(tinytorch::sum(c + a));
        EXPECT_FLOAT_EQ(a.grad().toFloat(), 2.0f) << "allocate_grad " << allocate_grad;
        EXPECT_FLOAT_EQ(b.grad().toFloat(), 1.0f) << "allocate_grad " << allocate_grad;

        tinytorch::backward(tinytorch::sum(a));
        EXPECT_FLOAT_EQ(a.grad().toFloat(), 3.0f) << "allocate_grad " << allocate_grad;
        EXPECT_FLOAT_EQ(b.grad().toFloat(), 1.0f) << "allocate_grad " << allocate_grad;
    }
}

// The implicit gradient of a single-element loss has the shape of the loss
TEST(TinyTorchAutograd, ImplicitGradientOfMultiDimLoss)
{
    Tensor a    = tttest::leaf({1, 2, 3}, {3, 1}, kCPU);
    Tensor loss = tinytorch::mean(a, 0, true);
    ASSERT_EQ(loss.sizes(), (SizeType{1, 1}));
    tinytorch::backward(loss);
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({1. / 3, 1. / 3, 1. / 3}, {3, 1}, kCPU), 1e-6, 1e-6);
}
