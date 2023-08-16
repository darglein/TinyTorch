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

struct SumNode : public FunctionNode<SumNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
         Tensor result = zeros({1}, a.options());
//        Tensor result = empty({1}, a.options());
        SELECT_DEVICE(a.device(), sum_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        CHECK_EQ(grad[0].numel(), 1);
        auto g        = grad[0];
        Tensor grad_a = empty(ctx->next_meta[0].size, g.options());
        SELECT_DEVICE(grad_a.device(), fill_impl, grad_a, g);
        return {grad_a};
    }
};
struct SumDimNode : public FunctionNode<SumDimNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue dim)
    {
        ctx->saved_data["dim"] = dim;
        auto out_size          = a.sizes();
        out_size[dim.toInt()]  = 1;
        Tensor result          = zeros(out_size, a.options());
        SELECT_DEVICE(result.device(), sum_impl, a, dim.toInt(), result);

        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        int dim       = ctx->saved_data["dim"].toInt();
        auto g        = grad[0];
        Tensor grad_a = empty(ctx->next_meta[0].size, g.options());
        SELECT_DEVICE(grad_a.device(), fill_impl, grad_a, g, dim);
        return {grad_a, {}};
    }
};
}  // namespace autograd

using namespace autograd;

Tensor pow(Tensor a, double b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), pow_impl, a, b, result);
    return result;
}
Tensor pow(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = empty_like(a);
    SELECT_DEVICE(a.device(), pow_impl, a, b, result);
    return result;
}

template <typename T>
static void fill_with_infinite(Tensor& a, bool positive_inf)
{
    double value;
    if (positive_inf)
    {
        value = 1000000;  // std::numeric_limits<T>::max();
    }
    else
    {
        value = -100000;  // std::numeric_limits<T>::lowest();
    }
    fill(a, value);
}



static void fill_with_infinite(Tensor& a, bool positive_inf)
{
    switch (a.scalar_type())
    {
        case kFloat16:
            fill_with_infinite<Half>(a, positive_inf);
            break;
        case kFloat32:
            fill_with_infinite<float>(a, positive_inf);
            break;
        case kFloat64:
            fill_with_infinite<double>(a, positive_inf);
            break;
        case kInt32:
            fill_with_infinite<int>(a, positive_inf);
            break;
        case kInt64:
            fill_with_infinite<int64_t>(a, positive_inf);
            break;
        default:
            CHECK(false);
    }
}

Tensor min(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());

    Tensor result = empty({1}, a.options());
    fill_with_infinite(result, true);
    SELECT_DEVICE(a.device(), min_impl, a, result);
    return result;
}
Tensor max(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    Tensor result = empty({1}, a.options());
    fill_with_infinite(result, false);
    SELECT_DEVICE(a.device(), max_impl, a, result);
    return result;
}
std::pair<Tensor, Tensor> min(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());

    auto result_size = a.sizes();
    result_size[dim] = 1;

    Tensor result = empty(result_size, a.options());
    fill_with_infinite(result, true);
    Tensor indices = empty(result_size, a.options().dtype(kLong));
    SELECT_DEVICE(a.device(), min_impl, a, dim, result, indices);

    if (!keepdim)
    {
        result = result.squeeze(dim);
        if (indices.defined()) indices = indices.squeeze(dim);
    }

    return {result, indices};
}
std::pair<Tensor, Tensor> max(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result_size = a.sizes();
    result_size[dim] = 1;

    Tensor result = empty(result_size, a.options());
    fill_with_infinite(result, false);
    Tensor indices = empty(result_size, a.options().dtype(kLong));

    SELECT_DEVICE(a.device(), max_impl, a, dim, result, indices);

    if (!keepdim)
    {
        result = result.squeeze(dim);
        if (indices.defined()) indices = indices.squeeze(dim);
    }

    return {result, indices};
}
Tensor min(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    Tensor result = empty_like(a);
    SELECT_DEVICE(a.device(), min_impl, a, b, result);
    return result;
}
Tensor max(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    Tensor result = empty_like(a);
    SELECT_DEVICE(a.device(), max_impl, a, b, result);
    return result;
}

Tensor sum(Tensor a)
{
    return SumNode::forward_and_build_graph(a)[0];
}

Tensor sum(Tensor a, int64_t dim, bool keepdim)
{
    CHECK_LT(dim, a.dim());
    auto result = autograd::SumDimNode::apply(a, dim)[0];
    if (!keepdim)
    {
        auto dims_before = result.dim();
        result           = result.squeeze(dim);
        CHECK_EQ(result.dim(), dims_before - 1);
    }
    return result;
}
Tensor sum(Tensor a, SizeType s, bool keepdim)
{
    auto sv = s.vec();
    std::sort(sv.begin(), sv.end(),std::greater<>());
    for (auto dim : sv)
    {
        a = sum(a, dim, keepdim);
    }
    return a;
}

Tensor mean(Tensor a)
{
    return sum(a) / (double)a.numel();  // TODO: This is not safe for small datatypes, which might overflow in the sum.
}

Tensor mean(Tensor a, int64_t dim, bool keepdim)
{
    auto count  = a.size(dim);
    auto result = sum(a, dim, keepdim);
    return result / (double)count;
}
Tensor mean(Tensor a, SizeType s, bool keepdim)
{
    auto sv = s.vec();
    std::sort(sv.begin(), sv.end(),std::greater<>());

    for (auto dim : sv)
    {
        a = mean(a, dim, keepdim);
    }

    return a;
}

Tensor std(Tensor a)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());

    Tensor result = empty({1}, a.options());
    cpu_impl::std_impl(a, result);
    return result;
}


Tensor std(Tensor a, int64_t dim)
{
    CHECK(false);
    return Tensor();
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
    SELECT_DEVICE(a.device(), clamp_impl_, a, low, high);
}

Tensor norm(Tensor a, int64_t norm, int64_t dim, bool keep)
{
    CHECK_EQ(norm, 2);
    a = a.square();
    a = a.sum(dim, keep);
    a = a.sqrt();
    return a;
}
Tensor prod(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto new_size = a.sizes();
    new_size[dim] = 1;
    auto result   = ones(new_size, a.options());
    SELECT_DEVICE(a.device(), prod_impl, a, dim, result);

    if (!keepdim)
    {
        result = result.squeeze(dim);
    }

    return result;
}
Tensor cumprod(Tensor a, int64_t dim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = ones_like(a);
    SELECT_DEVICE(a.device(), cumprod_impl, a, dim, result);
    return result;
}
Tensor cumsum(Tensor a, int64_t dim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    auto result = ones_like(a);
    SELECT_DEVICE(a.device(), cumsum_impl, a, dim, result);
    return result;
}

}  // namespace tinytorch