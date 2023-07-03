/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops.h"
#include "torch/cpu/ops_impl_cpu.h"

#include "graph.h"


namespace tinytorch
{

namespace autograd
{
struct SquareNode : public FunctionNode<SquareNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result   = square_impl_cpu(a);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        auto grad_a = square_backward_impl_cpu(l[0], grad[0]);
        return grad_a;
    }
};

struct AddNode : public FunctionNode<AddNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        auto result = add_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto grad_a = add_backward_impl_cpu(grad[0]);
        return grad_a;
    }
};

struct SubNode : public FunctionNode<SubNode>
{
    static std::vector<Tensor> forward(Context* ctx,  Tensor a, Tensor b)
    {
        auto result = sub_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto grad_a = sub_backward_impl_cpu(grad[0]);
        return grad_a;
    }
};


struct MultNode : public FunctionNode<MultNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        ctx->save_for_backward({a, b});
        auto result    = mult_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        auto grad_a = mult_backward_impl_cpu(l[0], l[1], grad[0]);
        return grad_a;
    }
};

struct SumNode : public FunctionNode<SumNode>
{
    static std::vector<Tensor> forward(Context* ctx,  Tensor a)
    {
        ctx->data_sizes["sizes"] = a.sizes();
        auto result             = sum_impl_cpu(a);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        assert(grad.size() == 1);
        auto grad_a = sum_backward_impl_cpu(ctx->data_sizes["sizes"], grad[0]);
        return grad_a;
    }
};
}

using namespace autograd;

Tensor square(Tensor a)
{
    return SquareNode::forward_and_build_graph(a)[0];
}

Tensor operator-(Tensor a, Tensor b)
{
    return SubNode::forward_and_build_graph(a, b)[0];
}

Tensor operator+(Tensor a, Tensor b)
{
    return AddNode::forward_and_build_graph(a, b)[0];
}

Tensor operator*(Tensor a, Tensor b)
{
    return MultNode::forward_and_build_graph(a, b)[0];
}
Tensor sum(Tensor a)
{
    return SumNode::forward_and_build_graph(a)[0];
}


void fill(Tensor& t, double value)
{
    fill_impl_cpu(t, value);
}

}  // namespace tinytorch