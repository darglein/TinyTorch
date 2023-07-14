/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "graph.h"
#include "torch/core/ops_math_functions.h"

#include "torch/cpu/ops_impl_cpu.h"


namespace tinytorch
{


namespace autograd
{
struct SquareNode : public FunctionNode<SquareNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = square_impl_cpu(a);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l      = ctx->get_saved_variables();
        auto grad_a = square_backward_impl_cpu(l[0], grad[0]);
        return grad_a;
    }
};


struct SumNode : public FunctionNode<SumNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->data_sizes["sizes"] = a.sizes();
        auto result              = sum_impl_cpu(a);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        auto grad_a = sum_backward_impl_cpu(ctx->data_sizes["sizes"], grad[0]);
        return grad_a;
    }
};
}  // namespace autograd

using namespace autograd;

Tensor square(Tensor a)
{
    return SquareNode::forward_and_build_graph(a)[0];
}

Tensor sum(Tensor a)
{
    return SumNode::forward_and_build_graph(a)[0];
}


Tensor min(Tensor a)
{
    CHECK(!a.requires_grad());
    return min_impl_cpu(a);
}
Tensor max(Tensor a)
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return max_impl_cpu(a);
}
std::pair<Tensor, Tensor> min(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return min_impl_cpu(a, dim, keepdim);
}
std::pair<Tensor, Tensor> max(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return max_impl_cpu(a, dim, keepdim);
}
Tensor min(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return min_impl_cpu(a, b);
}
Tensor max(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return max_impl_cpu(a, b);
}

Tensor mean(Tensor a)
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return sum(a) / (double)a.numel(); // TODO: This is not safe for small datatypes, which might overflow in the sum.
}

Tensor std(Tensor a)
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return std_impl_cpu(a);
}

Tensor abs(Tensor a) 
{
    CHECK(!a.requires_grad()|| !GradMode::is_enabled());
    return abs_impl_cpu(a);
}




}  // namespace tinytorch