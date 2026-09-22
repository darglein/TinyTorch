/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

// needed for the `torch::` namespace alias used inside the TORCH_MODULE macro
#include "torch/torch.h"

namespace
{

// A minimal module: one weight parameter registered at construction.
struct ToyImpl : public tinytorch::nn::Module
{
    Tensor w;

    explicit ToyImpl(Tensor weight)
        : w(weight)
    {
        register_parameter("w", w);
    }
};
// The macro expands to a class definition, so it must appear without a
// namespace qualifier; ToyImpl is found via unqualified lookup in this namespace.
TORCH_MODULE(Toy);

}  // namespace

// ------------------------------------------------------------------ nn::Module

TEST(Module, RegisterParameter)
{
    Tensor w = tttest::make_tensor({1, 2, 3, 4}, {4}, kCPU);
    ToyImpl m(w);

    EXPECT_EQ(m.parameters().size(), 1u);
    EXPECT_EQ(m.named_parameters().count("w"), 1u);

    // register_parameter allocates the gradient and sets requires_grad
    EXPECT_TRUE(m.named_parameters().at("w").requires_grad());
    EXPECT_TRUE(m.named_parameters().at("w").grad().defined());
    EXPECT_EQ(m.named_parameters().at("w").grad().numel(), 4);
}

TEST(Module, RegisterBuffer)
{
    ToyImpl m(tttest::make_tensor({1}, {1}, kCPU));

    Tensor b = tttest::make_tensor({9, 8}, {2}, kCPU);
    m.register_buffer("b", b);

    EXPECT_EQ(m.buffers().size(), 1u);
    EXPECT_EQ(m.named_parameters().size(), 1u);  // buffers are not parameters
}

TEST(Module, ZeroGrad)
{
    ToyImpl m(tttest::make_tensor({1, 2}, {2}, kCPU));
    m.parameters()[0].mutable_grad().fill_(5.0);

    m.zero_grad();
    TT_EXPECT_CLOSE(m.parameters()[0].grad(), tinytorch::zeros({2}, TensorOptions().device(kCPU)), 0, 0);

    m.parameters()[0].mutable_grad().fill_(5.0);
    m.zero_grad(true);  // set_to_none: the gradient becomes undefined
    EXPECT_FALSE(m.parameters()[0].grad().defined());
}

TEST(Module, SubModules)
{
    ToyImpl outer(tttest::make_tensor({1}, {1}, kCPU));
    auto inner = std::make_shared<ToyImpl>(tttest::make_tensor({2, 3}, {2}, kCPU));
    outer.register_module("inner", inner);

    EXPECT_EQ(outer.children().size(), 1u);
    EXPECT_EQ(outer.children()[0].get(), inner.get());

    // to() / zero_grad() / train() propagate to children
    outer.train(true);
    EXPECT_TRUE(inner->is_training());
}

TEST(Module, NameAndTraining)
{
    ToyImpl m(tttest::make_tensor({1}, {1}, kCPU));
    EXPECT_TRUE(m.name().size() > 0);
    EXPECT_TRUE(m.is_training());
}

// ------------------------------------------------------------------ ModuleHolder / TORCH_MODULE

TEST(ModuleHolder, PtrAndOperators)
{
    Tensor w = tttest::make_tensor({1, 2}, {2}, kCPU);
    Toy m(w);

    EXPECT_TRUE(static_cast<bool>(m));
    EXPECT_FALSE(m.is_empty());
    EXPECT_EQ(m.ptr()->parameters().size(), 1u);
    EXPECT_EQ(m->parameters().size(), 1u);
    EXPECT_EQ((*m).parameters().size(), 1u);
}

TEST(ModuleHolder, NullState)
{
    Toy m = nullptr;
    EXPECT_FALSE(static_cast<bool>(m));
    EXPECT_TRUE(m.is_empty());
    EXPECT_EQ(m.ptr(), nullptr);
}

// ------------------------------------------------------------------ to(device)

TEST(Module, ToDevice)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor w = tttest::make_tensor({1, 2, 3, 4}, {4}, kCPU);
    ToyImpl m(w);
    m.to(kCUDA);

    EXPECT_EQ(m.parameters()[0].device(), kCUDA);
    TT_EXPECT_CLOSE(m.parameters()[0], tttest::make_tensor({1, 2, 3, 4}, {4}, kCUDA), 0, 0);
}
