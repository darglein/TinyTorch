/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl.h"


namespace tinytorch
{


namespace autograd
{

struct AbsNode : public FunctionNode<AbsNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = empty_like(a);
        SELECT_DEVICE(a.device(), abs_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        auto a = l[0];
        auto g = grad[0];

        auto low    = a < 0;
        auto high   = a > 0;
        auto grad_a = (low * -1 + high) * g;

        return {grad_a};
    }
};
struct SqrtNode : public FunctionNode<SqrtNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = empty_like(a);
        SELECT_DEVICE(a.device(), sqrt_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto g      = grad[0];
        auto grad_a = 1 / (2 * a.sqrt()) * g;
        return {grad_a};
    }
};
struct LogNode : public FunctionNode<LogNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = empty_like(a);
        SELECT_DEVICE(a.device(), log_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto g      = grad[0];
        auto grad_a = 1 / a * g;
        return {grad_a};
    }
};
struct ExpNode : public FunctionNode<ExpNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = empty_like(a);
        SELECT_DEVICE(a.device(), exp_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto g      = grad[0];
        auto grad_a = exp(a) * g;
        return {grad_a};
    }
};

struct ReluNode : public FunctionNode<ReluNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = empty_like(a);
        SELECT_DEVICE(a.device(), relu_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto g      = grad[0];
        auto high   = a > 0;
        auto grad_a = high * g;
        return {grad_a};
    }
};
struct SigmoidNode : public FunctionNode<SigmoidNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = empty_like(a);
        SELECT_DEVICE(a.device(), sigmoid_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        auto a = l[0];
        auto g = grad[0];
        auto g_a = empty_like(a);
        SELECT_DEVICE(a.device(), sigmoid_backward_impl, a, g_a, g);
        return {g_a};
    }
};
struct SoftplusNode : public FunctionNode<SoftplusNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue beta)
    {
        ctx->save_for_backward({a});
        ctx->saved_data["beta"] = beta;
        auto result = empty_like(a);
        SELECT_DEVICE(a.device(), softplus_impl, a, beta.toDouble(), result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l = ctx->get_saved_variables();
        auto a = l[0];
        auto g = grad[0];
        auto g_a = empty_like(a);

        SELECT_DEVICE(a.device(), softplus_backward_impl, a, ctx->saved_data["beta"].toDouble(), g_a, g);
        return {g_a, {}};
    }
};
}  // namespace autograd

using namespace autograd;


Tensor abs(Tensor a)
{
    return autograd::AbsNode::apply(a)[0];
}

Tensor round(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), round_impl, a, result);
    return result;
}
Tensor square(Tensor a)
{
    return a * a;
}
Tensor sqrt(Tensor a)
{
    return SqrtNode::forward_and_build_graph(a)[0];
}
Tensor log(Tensor a)
{
    return LogNode::forward_and_build_graph(a)[0];
}
Tensor log1p(Tensor a)
{
    return log(1 + a);
}
Tensor exp(Tensor a)
{
    return ExpNode::forward_and_build_graph(a)[0];
}

Tensor sign(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), sign_impl, a, result);
    return result;
}
Tensor sin(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), sin_impl, a, result);
    return result;
}
Tensor cos(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), cos_impl, a, result);
    return result;
}

Tensor relu(Tensor a)
{
    return ReluNode::forward_and_build_graph(a)[0];
}
Tensor sigmoid(Tensor a)
{
    return SigmoidNode::forward_and_build_graph(a)[0];
}
Tensor softplus(Tensor a, double beta)
{
    return SoftplusNode::forward_and_build_graph(a, beta)[0];
}

}  // namespace tinytorch