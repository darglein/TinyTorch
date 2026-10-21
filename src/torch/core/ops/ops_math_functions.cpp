/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_math_functions.h"

#include "ops_impl.h"



namespace tinytorch
{

static int64_t normalize_dim(const Tensor& a, int64_t dim)
{
    int64_t n = a.dim();
    if (dim < 0) dim += n;
    CHECK(dim >= 0 && dim < n) << "dim out of range: " << dim << " for a tensor with " << n << " dims";
    return dim;
}

namespace autograd
{
struct AbsSumNode : public FunctionNode<AbsSumNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        Tensor result = zeros({1}, a.options());
        ctx->save_for_backward({a});
        SELECT_DEVICE(a.device(), abs_sum_impl, a, result);
        return {result};
    }

    // d/dx sum(|x|) = sign(x) * grad
    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        auto a        = ctx->get_saved_variables()[0];
        Tensor grad_a = sign(a) * grad[0];
        return {grad_a};
    }
};
struct ProdSumNode : public FunctionNode<ProdSumNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        Tensor result = zeros({1}, a.options());
        ctx->save_for_backward({a});
        SELECT_DEVICE(a.device(), prod_sum_impl, a, result);
        return {result};
    }

    // prod_sum(a) = sum(x^2)  ->  d/dx = 2*x*grad
    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        auto a        = ctx->get_saved_variables()[0];
        Tensor grad_a = (a * 2.0) * grad[0];
        return {grad_a};
    }
};
struct SumNode : public FunctionNode<SumNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        Tensor result = zeros({1}, a.options());
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

        Tensor result = zeros(out_size, a.options());
        if (a.numel() == 0)
        {
            return {result};
        }

        auto [a_collapsed, new_dim] = a.collapse_view(dim.toInt());
        CHECK_NE(new_dim, -1);
        auto out_size_collapsed     = a_collapsed.sizes();
        out_size_collapsed[new_dim] = 1;


        // CHECK(a_collapsed.dim() > 1) << a.sizes() << " " << a_collapsed.sizes();
        // SELECT_DEVICE(result.device(), sum_impl, a, dim.toInt(), result);
        if (a_collapsed.dim() == 1)
        {
            // global reduce
            SELECT_DEVICE(result.device(), sum_impl, a_collapsed, result.view(out_size_collapsed));
        }
        else
        {
            SELECT_DEVICE(result.device(), sum_impl, a_collapsed, new_dim, result.view(out_size_collapsed));
        }

        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        int64_t dim   = ctx->saved_data["dim"].toInt();
        auto g        = grad[0];
        Tensor grad_a = empty(ctx->next_meta[0].size, g.options());
        SELECT_DEVICE(grad_a.device(), fill_impl, grad_a, g, dim);
        return {grad_a, {}};
    }
};

struct PowNode : public FunctionNode<PowNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        auto [expand_a, expand_b]   = CheckOperatorSizeMatchOneDim(a, b);
        ctx->saved_data["expand_a"] = expand_a;
        ctx->saved_data["expand_b"] = expand_b;
        Tensor result               = empty(max_size(a, b), a.options());
        ctx->save_for_backward({a, b});
        SELECT_DEVICE(result.device(), pow_impl, a, b, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        auto a = ctx->get_saved_variables()[0];
        auto b = ctx->get_saved_variables()[1];
        auto g = grad[0];
        Tensor grad_a;
        Tensor grad_b;

        if (ctx->requires_grad_for_input(0))
        {
            // b * a^(b-1); where b == 0 the exponent is shifted to 0 so that a == 0 gives 0 instead of 0 * inf
            grad_a = empty(max_size(a, b), g.options());
            SELECT_DEVICE(grad_a.device(), pow_impl, a, b - 1 + (b == 0), grad_a);
            grad_a = grad_a * b * g;
        }

        if (ctx->requires_grad_for_input(1))
        {
            // ln(a)*a^b; like PyTorch 0 where a == 0 and b >= 0 (log(0) * 0^b would be NaN): log(1) = 0 there
            Tensor a_zero = (a == 0) * (1 - (b < 0));
            grad_b        = log(a + a_zero) * pow(a, b) * g;
        }

        BackwardExpand(grad_a, grad_b, ctx->saved_data["expand_a"].toSizes(), ctx->saved_data["expand_b"].toSizes());

        return {grad_a, grad_b};
    }
};

struct PowScalarNode : public FunctionNode<PowScalarNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue b)
    {
        ctx->saved_data["b"] = b;
        ctx->save_for_backward({a});

        auto result = empty_like(a);
        SELECT_DEVICE(result.device(), pow_impl, a, b.toDouble(), result);

        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        CHECK_EQ(grad.size(), 1);
        auto a   = ctx->get_saved_variables()[0];
        double b = ctx->saved_data["b"].toDouble();
        auto g   = grad[0];
        if (b == 0)
        {
            // a^0 is constant; b * a^(b-1) would be 0 * inf = NaN at a == 0
            return {zeros(ctx->next_meta[0].size, g.options()), {}};
        }
        Tensor grad_a = empty(ctx->next_meta[0].size, g.options());

        SELECT_DEVICE(grad_a.device(), pow_impl, a, b - 1, grad_a);
        grad_a = grad_a * g * b;
        return {grad_a, {}};
    }
};

}  // namespace autograd

using namespace autograd;

Tensor pow(Tensor a, double b)
{
    return autograd::PowScalarNode::apply(a, b)[0];
}
Tensor pow(Tensor a, Tensor b)
{
    return autograd::PowNode::apply(a, b)[0];
}

// Elementwise binary op without autograd: a numel-1 operand broadcasts to any shape, otherwise the dims must match
// and the sizes broadcast. Returns the result size.
static SizeType broadcast_operands(Tensor& a, Tensor& b)
{
    if (a.dim() != b.dim())
    {
        if (b.numel() == 1)
        {
            b = b.reshape(std::vector<int64_t>(a.dim(), 1));
        }
        else if (a.numel() == 1)
        {
            a = a.reshape(std::vector<int64_t>(b.dim(), 1));
        }
    }
    CheckOperatorSizeMatchOneDim(a, b);
    return max_size(a, b);
}

template <typename T>
static void fill_with_infinite(Tensor& a, bool positive_inf)
{
    double value;


    if constexpr (std::is_integral<T>::value)
    {
        // (double)max() of int64 rounds up to 2^63, which doesn't fit: use the largest double below it
        value = positive_inf ? std::nextafter((double)std::numeric_limits<T>::max(), 0.0)
                             : (double)std::numeric_limits<T>::lowest();
    }
    else
    {
        value = positive_inf ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
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
        case kUInt8:
            fill_with_infinite<uint8_t>(a, positive_inf);
            break;
        case kInt16:
            fill_with_infinite<int16_t>(a, positive_inf);
            break;
        case kUInt16:
            fill_with_infinite<uint16_t>(a, positive_inf);
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
    dim = normalize_dim(a, dim);
    CHECK_GT(a.size(dim), 0) << "min() over an empty dimension";

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
    dim = normalize_dim(a, dim);
    CHECK_GT(a.size(dim), 0) << "max() over an empty dimension";
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
    Tensor result = empty(broadcast_operands(a, b), a.options());
    SELECT_DEVICE(a.device(), min_impl, a, b, result);
    return result;
}
Tensor max(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    Tensor result = empty(broadcast_operands(a, b), a.options());
    SELECT_DEVICE(a.device(), max_impl, a, b, result);
    return result;
}


Tensor abs_sum(Tensor a)
{
    return AbsSumNode::forward_and_build_graph(a)[0];
}

TINYTORCH_API Tensor prod_sum(Tensor a)
{
    return ProdSumNode::forward_and_build_graph(a)[0];
}


Tensor sum(Tensor a)
{
    return SumNode::forward_and_build_graph(a)[0];
}

Tensor sum(Tensor a, int64_t dim, bool keepdim)
{
    dim = normalize_dim(a, dim);
    Tensor result;
    if (a.size(dim) != 1)
    {
        result = autograd::SumDimNode::apply(a, dim)[0];
    }
    else
    {
        result = a;
    }
    if (!keepdim)
    {
        result = result.squeeze(dim);
    }
    return result;
}
Tensor sum(Tensor a, SizeType s, bool keepdim)
{
    auto sv = s.vec();
    for (auto& d : sv) d = normalize_dim(a, d);
    std::sort(sv.begin(), sv.end(), std::greater<>());
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
    dim         = normalize_dim(a, dim);
    auto count  = a.size(dim);
    auto result = sum(a, dim, keepdim);
    return result / (double)count;
}
Tensor mean(Tensor a, SizeType s, bool keepdim)
{
    auto sv = s.vec();
    for (auto& d : sv) d = normalize_dim(a, d);
    std::sort(sv.begin(), sv.end(), std::greater<>());

    for (auto dim : sv)
    {
        a = mean(a, dim, keepdim);
    }

    return a;
}
Tensor median(Tensor a, double percentile)
{
    auto initial_device = a.device();
    auto result         = ones({1}, TensorOptions().dtype(a.scalar_type()).device(kCPU));
    cpu_impl::median_impl(a.cpu(), result, percentile);
    return result.to(initial_device);
}

Tensor std(Tensor a)
{
    auto mean = a.mean();

    Tensor result;
#ifdef TT_HAS_CUDA
    if (a.is_cuda() && (!a.requires_grad() || !GradMode::is_enabled()))
    {
        result = zeros_like(mean);
        cuda_impl::std_helper_impl(a, mean, result);
        result = result / (double)a.numel();
    }
    else
#endif
    {
        result = a - mean;
        result = result.square();
        result = result.mean();
    }
    result = result.sqrt();
    return result;
}


Tensor std(Tensor a, int64_t dim)
{
    // population std along dim, computed from differentiable ops so it works on any device
    dim       = normalize_dim(a, dim);
    auto mean = a.mean(dim, true);
    auto d    = a - mean;
    auto var  = d.square().mean(dim, true);
    return var.sqrt().squeeze(dim);
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
    dim = normalize_dim(a, dim);
    a   = a.square();
    a   = a.sum(dim, keep);
    a   = a.sqrt();
    return a;
}

void add_poisson_noise_(Tensor a)
{
    SELECT_DEVICE(a.device(), add_poisson_noise_impl, a);
}

Tensor prod(Tensor a, int64_t dim, bool keepdim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    dim           = normalize_dim(a, dim);
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
    dim         = normalize_dim(a, dim);
    auto result = ones_like(a);
    SELECT_DEVICE(a.device(), cumprod_impl, a, dim, result);
    return result;
}
Tensor cumsum(Tensor a, int64_t dim)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    dim         = normalize_dim(a, dim);
    auto result = ones_like(a);
    SELECT_DEVICE(a.device(), cumsum_impl, a, dim, result);
    return result;
}
}  // namespace tinytorch