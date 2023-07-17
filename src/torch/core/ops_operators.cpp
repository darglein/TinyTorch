/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_operators.h"

#include "graph.h"

#include "torch/cpu/ops_impl_cpu.h"
#include "torch/cpu/ops_operators_impl_cpu.h"

namespace tinytorch
{


std::ostream& operator<<(std::ostream& strm, Tensor t)
{
    print_impl_cpu(strm, t);
    return strm;
}



namespace autograd
{

struct AddNode : public FunctionNode<AddNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        CHECK_EQ(a.sizes(), b.sizes());
        auto result = empty_like(a);
        add_impl_cpu(a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto grad_a = grad[0].clone();
        auto grad_b = grad[0].clone();
        return {grad_a, grad_b};
    }
};

struct SubNode : public FunctionNode<SubNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        CHECK_EQ(a.sizes(), b.sizes());
        auto result = empty_like(a);
        sub_impl_cpu(a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto grad_a = grad[0].clone();
        auto grad_b = -grad[0].clone();
        return {grad_a, grad_b};
    }
};

struct DivNode : public FunctionNode<DivNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        CHECK_EQ(a.sizes(), b.sizes());
        auto result = empty_like(a);
        ctx->save_for_backward({a, b});
        div_impl_cpu(a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();

#if 0
        auto grad_a = empty_like(l[0]);
        auto grad_b = empty_like(l[1]);
        div_backward_impl_cpu(l[0], l[1], grad[0], grad_a, grad_b);
#else
        auto a = l[0];
        auto b = l[1];
        auto g = grad[0];
        auto grad_a = 1.0 / b * g;
        auto grad_b = -a / (b * b) * g;

#endif
        return {grad_a, grad_b};
    }
};

struct MultNode : public FunctionNode<MultNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        CHECK_EQ(a.sizes(), b.sizes());
        auto result = empty_like(a);
        ctx->save_for_backward({a, b});
        mult_impl_cpu(a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        auto a = l[0];
        auto b = l[1];
        auto g = grad[0];
        CHECK(!g.requires_grad());

        CHECK(!GradMode::is_enabled());
        auto grad_a = g * b;
        auto grad_b = g * a;
        CHECK(!grad_a.requires_grad());
        CHECK(!grad_b.requires_grad());
        return {grad_a, grad_b};
    }
};

struct DivScalarTensorNode : public FunctionNode<DivScalarTensorNode>
{
    static std::vector<Tensor> forward(Context* ctx, double a, Tensor b)
    {
        auto result          = empty_like(b);
        ctx->saved_data["a"] = a;
        ctx->save_for_backward({b});
        div_impl_cpu(a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        double a    = ctx->saved_data["a"].toDouble();
        auto l      = ctx->get_saved_variables();
        auto b = l[0];
        auto g = grad[0];

        auto grad_b = -a / (b * b) * g;
        // auto grad_b = empty_like(l[0]);
        // div_backward_impl_cpu(a, l[0], grad[0], grad_b);
        return {{}, grad_b};
    }
};

struct MultTensorScalarNode : public FunctionNode<MultTensorScalarNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, double b)
    {
        auto result          = empty_like(a);
        ctx->saved_data["b"] = b;
        ctx->save_for_backward({a});
        mult_impl_cpu(a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        double b    = ctx->saved_data["b"].toDouble();
        auto l      = ctx->get_saved_variables();
        auto grad_a = b * grad[0];
        return {grad_a, {}};
    }
};


struct AddTensorScalarNode : public FunctionNode<AddTensorScalarNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, double b)
    {
        auto result          = empty_like(a);
        ctx->saved_data["b"] = b;
        ctx->save_for_backward({a});

        add_impl_cpu(a, b, result);
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
    return DivScalarTensorNode::apply(a, b)[0];
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
Tensor operator+(double a, Tensor b)
{
    return b + a;
}


// ============================================================================

Tensor operator==(Tensor a, double b)
{
    auto result = empty_like(a);
    equal_impl_cpu(a, b, result);
    return result;
}

Tensor operator<(Tensor a, double b)
{
    auto result = empty_like(a);
    less_impl_cpu(a, b, result);
    return result;
}
Tensor operator>(Tensor a, double b)
{
    auto result = empty_like(a);
    greater_impl_cpu(a, b, result);
    return result;
}


Tensor operator+=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    add_impl_cpu(a, b, a);
    return a;
}
Tensor operator+=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    add_impl_cpu(a, b, a);
    return a;
}
Tensor operator-=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    sub_impl_cpu(a, b, a);
    return a;
}
Tensor operator*=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    mult_impl_cpu(a, b, a);
    return a;
}
Tensor operator*=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    mult_impl_cpu(a, b, a);
    return a;
}
Tensor operator/=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    div_impl_cpu(a, b, a);
    return a;
}


}  // namespace tinytorch