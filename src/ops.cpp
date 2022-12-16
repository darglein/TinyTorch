/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops.h"

#include "graph.h"


namespace tinytorch
{

struct SquareNode : public FunctionNode<SquareNode>
{
    static std::vector<Tensor> forward(Context& ctx, std::vector<Tensor> t)
    {
        ctx.data["t"] = t[0];
        auto result   = square_impl(t[0]);
        return {result};
    }

    static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor> grad)
    {
        auto grad_a = square_backward_impl(ctx.data["t"], grad[0]);
        return grad_a;
    }
};

struct AddNode : public FunctionNode<AddNode>
{
    static std::vector<Tensor> forward(Context& ctx, std::vector<Tensor> t)
    {
        auto result = add_impl(t[0], t[1]);
        return {result};
    }

    static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor> grad)
    {
        auto grad_a = add_backward_impl(grad[0]);
        return grad_a;
    }
};

struct SubNode : public FunctionNode<SubNode>
{
    static std::vector<Tensor> forward(Context& ctx, std::vector<Tensor> t)
    {
        auto result = sub_impl(t[0], t[1]);
        return {result};
    }

    static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor> grad)
    {
        auto grad_a = sub_backward_impl(grad[0]);
        return grad_a;
    }
};


struct MultNode : public FunctionNode<MultNode>
{
    static std::vector<Tensor> forward(Context& ctx, std::vector<Tensor> t)
    {
        ctx.data["t0"] = t[0];
        ctx.data["t1"] = t[1];
        auto result    = mult_impl(t[0], t[1]);
        return {result};
    }

    static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor> grad)
    {
        auto grad_a = mult_backward_impl(ctx.data["t0"], ctx.data["t1"], grad[0]);
        return grad_a;
    }
};

struct SumNode : public FunctionNode<SumNode>
{
    static std::vector<Tensor> forward(Context& ctx, std::vector<Tensor> t)
    {
        ctx.data_int["size"] = t[0].size();
        auto result          = sum_impl(t[0]);
        return {result};
    }

    static std::vector<Tensor> backward(Context& ctx, std::vector<Tensor> grad)
    {
        assert(grad.size() == 1);
        auto grad_a = sum_backward_impl(ctx.data_int["size"], grad[0]);
        return grad_a;
    }
};



Tensor square(Tensor a)
{
    return SquareNode::forward_and_build_graph({a})[0];
}

Tensor operator-(Tensor a, Tensor b)
{
    return SubNode::forward_and_build_graph({a, b})[0];
}

Tensor operator+(Tensor a, Tensor b)
{
    return AddNode::forward_and_build_graph({a, b})[0];
}

Tensor operator*(Tensor a, Tensor b)
{
    return MultNode::forward_and_build_graph({a, b})[0];
}
Tensor sum(Tensor a)
{
    return SumNode::forward_and_build_graph({a})[0];
}


}  // namespace tinytorch