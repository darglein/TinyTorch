/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_operators.h"

#include "graph.h"

#include "torch/cpu/ops_impl_cpu.h"

namespace tinytorch
{
namespace autograd
{

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
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
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

struct DivNode : public FunctionNode<DivNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        ctx->save_for_backward({a, b});
        auto result = div_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        return div_backward_impl_cpu(l[0], l[1], grad[0]);
    }
};

struct MultNode : public FunctionNode<MultNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        ctx->save_for_backward({a, b});
        auto result = mult_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l      = ctx->get_saved_variables();
        auto grad_a = mult_backward_impl_cpu(l[0], l[1], grad[0]);
        return grad_a;
    }
};

struct DivScalarTensorNode : public FunctionNode<DivScalarTensorNode>
{
    static std::vector<Tensor> forward(Context* ctx, double a, Tensor b)
    {
        ctx->saved_data["a"] = a;
        ctx->save_for_backward({b});
        auto result = div_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        double a    = ctx->saved_data["a"].toDouble();
        auto l      = ctx->get_saved_variables();
        auto grad_a = div_backward_impl_cpu(a, l[0], grad[0]);
        return {{}, grad_a[0]};
    }
};

struct MultTensorScalarNode : public FunctionNode<MultTensorScalarNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, double b)
    {
        ctx->saved_data["b"] = b;
        ctx->save_for_backward({a});
        auto result = mult_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        double b    = ctx->saved_data["b"].toDouble();
        auto l      = ctx->get_saved_variables();
        auto grad_a = mult_backward_impl_cpu(l[0], b, grad[0]);
        return {grad_a[0], {}};
    }
};


struct AddTensorScalarNode : public FunctionNode<AddTensorScalarNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, double b)
    {
        ctx->saved_data["b"] = b;
        ctx->save_for_backward({a});

        auto result = add_impl_cpu(a, b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        double b    = ctx->saved_data["b"].toDouble();
        auto l      = ctx->get_saved_variables();
        auto grad_a = grad[0].clone();
        return {grad_a, {}};
    }
};
}  // namespace autograd

using namespace autograd;

Tensor operator/(Tensor a, Tensor b)
{
    return DivNode::apply(a, b)[0];
}

Tensor operator/(Tensor a, double b)
{
    return a * (1.0 / b);
}

Tensor operator/(double a, Tensor b)
{
    return DivScalarTensorNode::apply(a,b)[0];
}


Tensor operator-(Tensor a, Tensor b)
{
    return SubNode::forward_and_build_graph(a, b)[0];
}
Tensor operator-(Tensor b)
{
    return b * (-1);
}
Tensor operator-(Tensor a, double b)
{
    return a + (-b);
}
Tensor operator-(double a, Tensor b)
{
    return a + (-b);
}


Tensor operator+(Tensor a, Tensor b)
{
    return AddNode::forward_and_build_graph(a, b)[0];
}

Tensor operator*(Tensor a, Tensor b)
{
    return MultNode::forward_and_build_graph(a, b)[0];
}



Tensor operator*(double a, Tensor b)
{
    return MultTensorScalarNode::forward_and_build_graph(b, a)[0];
}
Tensor operator*(Tensor a, double b)
{
    return b * a;
}

Tensor operator+(Tensor a, double b)
{
    return AddTensorScalarNode::forward_and_build_graph(a, b)[0];
}
}  // namespace tinytorch