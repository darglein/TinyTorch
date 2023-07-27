/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_operators.h"

#include "graph.h"

#include "torch/core/ops_impl.h"

namespace tinytorch
{


std::ostream& operator<<(std::ostream& strm, Tensor t)
{
    cpu_impl::print_impl(strm, t);
    return strm;
}


inline void CheckOperatorSizeMatchOneDim(const Tensor& a, const Tensor& b)
{
    CHECK_EQ(a.device(), b.device());
    CHECK_EQ(a.dim(), b.dim());
    int num_missmatch = 0;
    for (int i = 0; i < a.dim(); ++i)
    {
        if (a.size(i) != b.size(i))
        {
            CHECK(a.size(i) == 1 || b.size(i) == 1) << "Size Missmatch " << a.sizes() << " " << b.sizes();
            num_missmatch++;
        }
    }
    CHECK_LE(num_missmatch, 1) << "only one dim can be expanded during operator " << a.sizes() << " " << b.sizes();
}

// Operators can have the case that one Tensor is dimension 1 along one axis and the other is not.
// This computes the size of the result tensor and checks if everything else is ok.
static SizeType max_size(Tensor a, Tensor b)
{
    CHECK_EQ(a.dim(), b.dim());
    SizeType new_sizes;
    new_sizes.resize(a.dim());
    for (int64_t i = 0; i < a.dim(); ++i)
    {
        int64_t as = a.size(i);
        int64_t bs = b.size(i);
        CHECK(as == bs || as == 1 || bs == 1);
        new_sizes[i] = std::max(as, bs);
    }
    return new_sizes;
}


namespace autograd
{

struct AddNode : public FunctionNode<AddNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        CheckOperatorSizeMatchOneDim(a, b);
        Tensor result = empty(max_size(a, b), a.options());
        SELECT_DEVICE(a.device(), add_impl, a, b, result);
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
        CheckOperatorSizeMatchOneDim(a, b);
        Tensor result = empty(max_size(a, b), a.options());
        SELECT_DEVICE(a.device(), sub_impl, a, b, result);
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
        CheckOperatorSizeMatchOneDim(a, b);
        Tensor result = empty(max_size(a, b), a.options());
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

        return {grad_a, grad_b};
    }
};

struct MultNode : public FunctionNode<MultNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        CheckOperatorSizeMatchOneDim(a, b);
        Tensor result = empty(max_size(a, b), a.options());
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
    SELECT_DEVICE(a.device(), equal_impl, a, b, result);
    return result;
}

Tensor operator<(Tensor a, double b)
{
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), less_impl, a, b, result);
    return result;
}
Tensor operator>(Tensor a, double b)
{
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), greater_impl, a, b, result);
    return result;
}

Tensor operator+=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());

    SELECT_DEVICE(a.device(), add_impl, a, b, a);
    return a;
}
Tensor operator+=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(a.device(), add_impl, a, b, a);
    return a;
}
Tensor operator-=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(a.device(), sub_impl, a, b, a);
    return a;
}
Tensor operator-=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(a.device(), sub_impl, a, b, a);
    return a;
}
Tensor operator*=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(a.device(), mult_impl, a, b, a);
    return a;
}
Tensor operator*=(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(a.device(), mult_impl, a, b, a);
    return a;
}
Tensor operator/=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(a.device(), div_impl, a, b, a);
    return a;
}


}  // namespace tinytorch