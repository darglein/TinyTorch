/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_operators.h"

#include "torch/core/ops/ops_impl.h"

namespace tinytorch
{


std::ostream& operator<<(std::ostream& strm, Tensor t)
{
    CHECK(t.defined());
    cpu_impl::print_impl(strm, t.cpu());
    return strm;
}


namespace autograd
{

struct AddNode : public FunctionNode<AddNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        auto [expand_a, expand_b]   = CheckOperatorSizeMatchOneDim(a, b);
        ctx->saved_data["expand_a"] = expand_a;
        ctx->saved_data["expand_b"] = expand_b;
        Tensor result               = empty(max_size(a, b), a.options());
        SELECT_DEVICE(a.device(), add_impl, a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto grad_a = grad[0];
        auto grad_b = grad[0];
        BackwardExpand(grad_a, grad_b, ctx->saved_data["expand_a"].toSizes(), ctx->saved_data["expand_b"].toSizes());

        return {grad_a, grad_b};
    }
};

struct SubNode : public FunctionNode<SubNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        auto [expand_a, expand_b]   = CheckOperatorSizeMatchOneDim(a, b);
        ctx->saved_data["expand_a"] = expand_a;
        ctx->saved_data["expand_b"] = expand_b;
        Tensor result               = empty(max_size(a, b), a.options());
        SELECT_DEVICE(a.device(), sub_impl, a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        Tensor grad_a, grad_b;

        grad_a = grad[0];
        if (ctx->requires_grad_for_input(1))
        {
            grad_b = -grad[0];
        }

        BackwardExpand(grad_a, grad_b, ctx->saved_data["expand_a"].toSizes(), ctx->saved_data["expand_b"].toSizes());
        return {grad_a, grad_b};
    }
};

struct DivNode : public FunctionNode<DivNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)

    {
        auto [expand_a, expand_b]   = CheckOperatorSizeMatchOneDim(a, b);
        ctx->saved_data["expand_a"] = expand_a;
        ctx->saved_data["expand_b"] = expand_b;
        Tensor result               = empty(max_size(a, b), a.options());
        ctx->save_for_backward({a, b});
        SELECT_DEVICE(a.device(), div_impl, a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();

        auto a      = l[0];
        auto b      = l[1];
        auto g      = grad[0];
        auto grad_a = 1.0 / b * g;
        auto grad_b = -a / (b * b) * g;


        BackwardExpand(grad_a, grad_b, ctx->saved_data["expand_a"].toSizes(), ctx->saved_data["expand_b"].toSizes());

        return {grad_a, grad_b};
    }
};

struct MultNode : public FunctionNode<MultNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        auto [expand_a, expand_b]   = CheckOperatorSizeMatchOneDim(a, b);
        ctx->saved_data["expand_a"] = expand_a;
        ctx->saved_data["expand_b"] = expand_b;
        Tensor result               = empty(max_size(a, b), a.options());
        ctx->save_for_backward({a, b});
        SELECT_DEVICE(a.device(), mult_impl, a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        auto a = l[0];
        auto b = l[1];
        auto g = grad[0];
        Tensor grad_a, grad_b;
        if (ctx->requires_grad_for_input(0))
        {
            grad_a = g * b;
        }
        if (ctx->requires_grad_for_input(1))
        {
            grad_b = g * a;
        }
        BackwardExpand(grad_a, grad_b, ctx->saved_data["expand_a"].toSizes(), ctx->saved_data["expand_b"].toSizes());
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
        SELECT_DEVICE(b.device(), div_impl, a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        double a = ctx->saved_data["a"].toDouble();
        auto l   = ctx->get_saved_variables();
        auto b   = l[0];
        auto g   = grad[0];

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

        SELECT_DEVICE(a.device(), mult_impl, a, b, result);
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

        SELECT_DEVICE(a.device(), add_impl, a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        double b    = ctx->saved_data["b"].toDouble();
        auto l      = ctx->get_saved_variables();
        auto grad_a = grad[0];
        return {grad_a, {}};
    }
};
}  // namespace autograd

using namespace autograd;

inline void MatchTensorSize(Tensor& a, Tensor& b)
{
    if (b.dim() == 1 && b.size(0) == 1 && a.dim() > 1)
    {
        // expand b with enough dims to match a
        for (int i = 0; i < a.dim() - 1; ++i)
        {
            b = b.unsqueeze(0);
        }
    }
}

Tensor operator/(Tensor a, Tensor b)
{
    MatchTensorSize(a, b);
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
    MatchTensorSize(a, b);
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
    MatchTensorSize(a, b);
    return AddNode::forward_and_build_graph(a, b)[0];
}

Tensor operator*(Tensor a, Tensor b)
{
    MatchTensorSize(a, b);
    return MultNode::forward_and_build_graph(a, b)[0];
}



Tensor operator*(double a, Tensor b)
{
    return MultTensorScalarNode::forward_and_build_graph(b, a)[0];
}
Tensor operator*(Tensor a, double b)
{
    if (b == 1)
    {
        return a;
    }
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
{  CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), equal_impl, a, b, result);
    return result;
}

Tensor operator<(Tensor a, double b)
{  CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), less_impl, a, b, result);
    return result;
}
Tensor operator>(Tensor a, double b)
{  CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), greater_impl, a, b, result);
    return result;
}

Tensor operator+=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(a.device(), add_impl, a, b, a);
    return a;
}
Tensor operator+=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(a.device(), add_impl, a, b, a);
    return a;
}
Tensor operator-=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    MatchTensorSize(a, b);
    SELECT_DEVICE(a.device(), sub_impl, a, b, a);
    return a;
}
Tensor operator-=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(a.device(), sub_impl, a, b, a);
    return a;
}
Tensor operator*=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    MatchTensorSize(a, b);
    SELECT_DEVICE(a.device(), mult_impl, a, b, a);
    return a;
}
Tensor operator*=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(a.device(), mult_impl, a, b, a);
    return a;
}
Tensor operator/=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    MatchTensorSize(a, b);
    SELECT_DEVICE(a.device(), div_impl, a, b, a);
    return a;
}
Tensor operator/=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    return a *= (1.0 / b);
}

}  // namespace tinytorch