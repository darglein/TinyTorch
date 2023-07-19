/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops_math_functions.h"

#include "graph.h"

#include "torch/cpu/ops_impl_cpu.h"


namespace tinytorch
{


namespace autograd
{

struct SqrtNode : public FunctionNode<SqrtNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        ctx->save_for_backward({a});
        auto result = empty_like(a);
        sqrt_impl_cpu(a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto g      = grad[0];
        auto grad_a = 1 / (2*a.sqrt()) * g;
        return {grad_a};
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
    return a * a;
}


Tensor sqrt(Tensor a)
{
    return SqrtNode::forward_and_build_graph(a)[0];
}



Tensor min(Tensor a)
{
    CHECK(!a.requires_grad());
    return min_impl_cpu(a);
}
Tensor max(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return max_impl_cpu(a);
}
std::pair<Tensor, Tensor> min(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return min_impl_cpu(a, dim, keepdim);
}
std::pair<Tensor, Tensor> max(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return max_impl_cpu(a, dim, keepdim);
}
Tensor min(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return min_impl_cpu(a, b);
}
Tensor max(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return max_impl_cpu(a, b);
}

Tensor sum(Tensor a)
{
    return SumNode::forward_and_build_graph(a)[0];
}

Tensor sum(Tensor a, int64_t dim, bool squeeze_dim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());

    auto out_size = a.sizes();
    out_size[dim] = 1;
    auto result   = empty(out_size, a.options());
    sum_impl_cpu(a, dim, result);
    if (squeeze_dim)
    {
        auto dims_before = result.dim();
        result           = result.squeeze(dim);
        CHECK_EQ(result.dim(), dims_before - 1);
    }
    return result;
}
Tensor sum(Tensor a, SizeType s)
{
    for (auto dim : s.vec())
    {
        a = sum(a, dim, false);
    }
    return a;
}

Tensor mean(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return sum(a) / (double)a.numel();  // TODO: This is not safe for small datatypes, which might overflow in the sum.
}

Tensor mean(Tensor a, int64_t dim, bool squeeze_dim)
{
    auto count  = a.size(dim);
    auto result = sum(a, dim, squeeze_dim);
    return result / count;
}
Tensor mean(Tensor a, SizeType s)
{
    for (auto dim : s.vec())
    {
        a = mean(a, dim, false);
    }
    return a;
}

Tensor std(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return std_impl_cpu(a);
}


Tensor std(Tensor a, int64_t dim)
{
    return Tensor();
}


Tensor abs(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return abs_impl_cpu(a);
}
Tensor clamp(Tensor a, double low, double high)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto t = a.clone();
    clamp_(t, low, high);
    return t;
}
void clamp_(Tensor& a, double low, double high)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    CHECK(a.is_cpu());
    clamp_impl_cpu_(a, low, high);
}

Tensor norm(Tensor a, int64_t norm, int64_t dim, bool keep)
{
    CHECK_EQ(norm, 2);
    a = a.square();
    a = a.sum(dim, !keep);
    a = a.sqrt();
    return a;
}


}  // namespace tinytorch