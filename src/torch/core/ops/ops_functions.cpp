/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_functions.h"

#include "ops_impl.h"



namespace tinytorch
{

static void fill_neg_one_dim(SizeType& new_sizes, int64_t old_numel)
{
    int64_t new_numel = 1;
    int64_t* neg_dim  = nullptr;
    for (int64_t& i : new_sizes.vec())
    {
        CHECK_GE(i, -1) << "invalid shape " << new_sizes;
        if (i == -1)
        {
            CHECK_EQ(neg_dim, nullptr) << "only one dimension can be inferred";
            neg_dim = &i;
        }
        else
        {
            new_numel *= i;
        }
    }

    if (neg_dim)
    {
        CHECK_NE(new_numel, 0) << "cannot reshape a tensor of " << old_numel << " elements into shape " << new_sizes
                               << " because the unspecified dimension size -1 can be any value and is ambiguous";
        CHECK_EQ(old_numel % new_numel, 0);
        *neg_dim = old_numel / new_numel;
        new_numel *= *neg_dim;
    }

    CHECK_EQ(old_numel, new_numel);
}


namespace autograd
{
struct ReshapeNode : public FunctionNode<ReshapeNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue new_sizes_)
    {
        NoGradGuard ngg;
        ctx->saved_data["old_sizes"] = a.sizes();
        SizeType new_sizes           = new_sizes_.toSizes();
        fill_neg_one_dim(new_sizes, a.numel());

        if (a.is_contiguous())
        {
            return {a.view(new_sizes)};
        }

        Tensor result = empty(new_sizes, a.options());
        tinytorch::copy(a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        SizeType old_sizes = ctx->saved_data["old_sizes"].toSizes();
        auto g             = grad[0];

        if (g.is_contiguous())
        {
            return {g.view(old_sizes), {}};
        }

        Tensor grad_a = empty(old_sizes, g.options());
        tinytorch::copy(g, grad_a);
        return {grad_a, {}};
    }
};
}  // namespace autograd


Tensor reshape(Tensor t, SizeType sizes)
{
    if (t.sizes() == sizes)
    {
        return t;
    }
    return autograd::ReshapeNode::apply(t, sizes)[0];
}

Tensor repeat(Tensor t, SizeType repeats)
{
    CHECK_EQ(t.dim(), repeats.size());



    auto new_sizes = t.sizes();
    for (int i = 0; i < t.dim(); ++i)
    {
        new_sizes[i] *= repeats[i];
    }

    if (t.sizes() == new_sizes)
    {
        return t.clone();
    }
    CHECK(!t.requires_grad() || !GradMode::is_enabled());

    auto result = empty(new_sizes, t.options());
    SELECT_DEVICE(t.device(), repeat_impl, t, repeats, result);



    return result;
}

Tensor repeat_interleave(Tensor t, int64_t count)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    CHECK_GE(t.dim(), 1) << "repeat_interleave requires at least one dimension";
    CHECK_GE(count, 0);
    SizeType new_sizes = t.sizes();
    new_sizes[0] *= count;
    Tensor result = empty(new_sizes, t.options());

    SELECT_DEVICE(t.device(), repeat_interleave_impl, t, count, result);
    return result;
}

namespace autograd
{
struct TransposeNode : public FunctionNode<TransposeNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor t, IValue dim0, IValue dim1)
    {
        ctx->saved_data["dim0"] = dim0;
        ctx->saved_data["dim1"] = dim1;

        SizeType new_sizes = t.sizes();
        std::swap(new_sizes[dim0.toInt()], new_sizes[dim1.toInt()]);

        Tensor result = empty(new_sizes, t.options());
        SELECT_DEVICE(t.device(), transpose_impl, t, dim0.toInt(), dim1.toInt(), result);

        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int64_t dim0 = ctx->saved_data["dim0"].toInt();
        int64_t dim1 = ctx->saved_data["dim1"].toInt();

        auto g = grad[0];

        // The derivative of a transpose is simply transposing the gradient back
        // across the exact same two dimensions.
        SizeType grad_sizes = g.sizes();
        std::swap(grad_sizes[dim0], grad_sizes[dim1]);

        Tensor grad_t = empty(grad_sizes, g.options());
        SELECT_DEVICE(g.device(), transpose_impl, g, dim0, dim1, grad_t);

        // We return empty bracket lists {} for the two IValue inputs (dim0, dim1)
        return {grad_t, {}, {}};
    }
};
}  // namespace autograd

Tensor transpose(Tensor t, int64_t dim0, int64_t dim1)
{
    dim0 = maybe_wrap_dim(dim0, t.dim());
    dim1 = maybe_wrap_dim(dim1, t.dim());
    return autograd::TransposeNode::apply(t, dim0, dim1)[0];
}

void transpose(Tensor src, Tensor dst, int64_t dim0, int64_t dim1)
{
    CHECK(!src.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    CHECK_EQ(src.dim(), dst.dim());
    dim0 = maybe_wrap_dim(dim0, src.dim());
    dim1 = maybe_wrap_dim(dim1, src.dim());
    CHECK_EQ(src.size(dim0), dst.size(dim1));
    CHECK_EQ(src.size(dim1), dst.size(dim0));
    SELECT_DEVICE(src.device(), transpose_impl, src, dim0, dim1, dst);
}

void fill(Tensor& t, double value)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(t.device(), fill_impl, t, value);
}

void fill(Tensor& t, Tensor value)
{
    CHECK_EQ(value.numel(), 1);
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(t.device(), fill_impl, t, value);
}
void fill(Tensor& t, Tensor values, int dim)
{
    dim = (int)maybe_wrap_dim(dim, t.dim());
    CHECK_EQ(values.numel(), t.size(dim));
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(t.device(), fill_impl, t, values, dim);
}

void uniform(Tensor& t, double mi, double ma)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(t.device(), uniform_impl, t, mi, ma);
}

void uniform_int(Tensor& t, int64_t low, int64_t high)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    CHECK_LT(low, high);
    // a floating dtype must represent every value in [low, high) exactly, otherwise rounding can produce high
    int64_t max_exact = -1;
    switch (t.scalar_type())
    {
        case kHalf:
            max_exact = int64_t(1) << 11;
            break;
        case kFloat:
            max_exact = int64_t(1) << 24;
            break;
        case kDouble:
            max_exact = int64_t(1) << 53;
            break;
        default:
            break;
    }
    if (max_exact > 0)
    {
        CHECK_GE(low, -max_exact) << "randint: low is out of the exactly representable range of " << t.scalar_type();
        CHECK_LE(high - 1, max_exact) << "randint: high is out of the exactly representable range of "
                                      << t.scalar_type();
    }
    SELECT_DEVICE(t.device(), uniform_int_impl, t, low, high);
}

void normal_random(Tensor& t)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(t.device(), normal_random_impl, t);
}


void copy(Tensor src, Tensor target, bool async)
{
    CHECK(!src.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();

    if (src.device() == target.device())
    {
        SELECT_DEVICE(src.device(), copy_and_convert_impl, src, target);
    }
    else
    {
        cpu_impl::to_impl_cpu_cuda(src, target, async);
    }
}



namespace autograd
{
struct ToDeviceNode : public FunctionNode<ToDeviceNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue new_device_, IValue non_blocking)
    {
        NoGradGuard ngg;
        Device new_device = new_device_.toDevice();
#ifdef TT_HAS_CUDA
        Tensor contig = a.contiguous();
        Tensor result = empty(contig.sizes(), a.options().device(new_device).pinned_memory(false));
        cpu_impl::to_impl_cpu_cuda(contig, result, non_blocking.toBool());
        return {result};
#else
        CHECK(false);
#endif
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        Device old_device = ctx->next_meta[0].device;
        Tensor g          = grad[0].contiguous();
        Tensor grad_a     = empty(g.sizes(), g.options().device(old_device).pinned_memory(false));
        cpu_impl::to_impl_cpu_cuda(g, grad_a, false);
        return {grad_a, {}, {}};
    }
};

struct ToScalarTypeNode : public FunctionNode<ToScalarTypeNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue new_dtype_)
    {
        NoGradGuard ngg;
        ctx->saved_data["old_dtype"] = (int)a.dtype();
        Dtype new_dtype              = (Dtype)new_dtype_.toInt();

        Tensor result = empty_like(a, a.options().dtype(new_dtype));

        SELECT_DEVICE(a.device(), copy_and_convert_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        Dtype old_dtype = (Dtype)ctx->saved_data["old_dtype"].toInt();
        auto g          = grad[0];
        Tensor grad_a   = empty_like(grad[0], grad[0].options().dtype(old_dtype));
        SELECT_DEVICE(g.device(), copy_and_convert_impl, g, grad_a);
        return {grad_a, {}};
    }
};
}  // namespace autograd


Tensor to(Tensor a, Device new_device, bool non_blocking)
{
    if (a.device() == new_device)
    {
        return a;
    }

    return autograd::ToDeviceNode::apply(a.contiguous(), new_device, non_blocking)[0];
}

Tensor to(Tensor a, ScalarType other_type, bool non_blocking)
{
    if (a.dtype() == other_type)
    {
        return a;
    }
    return autograd::ToScalarTypeNode::apply(a, (int)other_type)[0];
}


namespace autograd
{
struct IndexSelectNode : public FunctionNode<IndexSelectNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor input, IValue dim, Tensor index)
    {
        if (ctx->requires_grad)
        {
            ctx->saved_data["dim"] = dim;
            ctx->save_for_backward({index});
        }


        CHECK_LT(dim.toInt(), input.dim());
        CHECK(index.dtype() == kInt32 || index.dtype() == kInt64);
        CHECK_EQ(index.dim(), 1);



        CHECK(!input.requires_grad() || !GradMode::is_enabled());

        auto result_size         = input.sizes().vec();
        result_size[dim.toInt()] = index.numel();
        Tensor result            = empty(result_size, input.options());


        SELECT_DEVICE(result.device(), index_select_impl, input, dim.toInt(), index, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int64_t dim      = ctx->saved_data["dim"].toInt();
        auto l           = ctx->get_saved_variables();
        auto index       = l[0];
        auto input_sizes = ctx->next_meta[0].size;

        auto g = grad[0];

        auto grad_input = zeros(input_sizes, g.options());

        grad_input = index_add(grad_input, dim, index, g);
        return {grad_input, {}, {}};
    }
};
}  // namespace autograd


Tensor index_select(Tensor input, int64_t dim, Tensor index)
{
    dim = maybe_wrap_dim(dim, input.dim());
    return autograd::IndexSelectNode::apply(input, dim, index)[0];
}

namespace autograd
{
struct IndexAddNode : public FunctionNode<IndexAddNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor input, IValue dim, Tensor index, Tensor data)
    {
        ctx->saved_data["dim"] = dim;
        ctx->save_for_backward({index});


        CHECK_LT(dim.toInt(), input.dim());
        CHECK(index.dtype() == kInt32 || index.dtype() == kInt64);
        CHECK_EQ(data.device(), input.device());
        CHECK_EQ(index.device(), input.device());
        CHECK_EQ(input.dim(), data.dim());
        CHECK_EQ(index.dim(), 1);
        CHECK_EQ(index.numel(), data.size(dim.toInt())) << index.sizes() << " " << data.sizes();
        for (int64_t d = 0; d < data.dim(); ++d)
        {
            CHECK(d == dim.toInt() || data.size(d) <= input.size(d))
                << "index_add: data " << data.sizes() << " does not fit into input " << input.sizes();
        }

        Tensor result = input.clone();
        SELECT_DEVICE(result.device(), index_add_impl, dim.toInt(), index, data, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int64_t dim = ctx->saved_data["dim"].toInt();
        auto l      = ctx->get_saved_variables();
        auto index  = l[0];

        auto grad_data  = index_select(grad[0], dim, index);
        auto grad_input = grad[0];

        // Tensor grad_a = empty_like(grad[0]);
        // copy_impl(grad[0], grad_a);
        return {grad_input, {}, {}, grad_data};
    }
};
}  // namespace autograd

Tensor index_add(Tensor input, int64_t dim, Tensor index, Tensor data)
{
    dim = maybe_wrap_dim(dim, input.dim());
    return autograd::IndexAddNode::apply(input, dim, index, data)[0];
}

void index_copy(Tensor& target, int64_t dim, Tensor index, Tensor source)
{
    CHECK(!target.requires_grad() || !GradMode::is_enabled());
    CHECK_EQ(target.dtype(), source.dtype());
    dim = maybe_wrap_dim(dim, target.dim());
    CHECK_EQ(target.device(), source.device());
    CHECK_EQ(index.device(), target.device());
    CHECK_EQ(target.dim(), source.dim());
    CHECK_EQ(index.dim(), 1);
    CHECK_EQ(index.numel(), source.size(dim)) << index.sizes() << " " << source.sizes();
    for (int64_t d = 0; d < source.dim(); ++d)
    {
        CHECK(d == dim || source.size(d) <= target.size(d))
            << "index_copy: source " << source.sizes() << " does not fit into target " << target.sizes();
    }
    TINYTORCH_LOG_FUNCTION_CALL();
    SELECT_DEVICE(target.device(), index_copy_impl, target, dim, index, source);
}

Tensor gather(Tensor data, int64_t dim, Tensor index)
{
    CHECK(!data.requires_grad() || !GradMode::is_enabled());
    TINYTORCH_LOG_FUNCTION_CALL();
    CHECK_EQ(data.dim(), index.dim());
    CHECK_EQ(index.dtype(), kLong) << "gather: index must be int64";
    CHECK_EQ(data.device(), index.device());
    dim = maybe_wrap_dim(dim, data.dim());
    for (int64_t d = 0; d < data.dim(); ++d)
    {
        CHECK(d == dim || index.size(d) <= data.size(d))
            << "gather: index " << index.sizes() << " is larger than data " << data.sizes() << " in dim " << d;
    }
    auto out_sizes = index.sizes();
    auto result    = empty(out_sizes, data.options());
    SELECT_DEVICE(result.device(), gather_impl, data, dim, index, result);
    return result;
}
void padding_2d(Tensor data, Tensor result, int pad_left, int pad_right, int pad_top, int pad_bottom, PaddingMode mode)
{
    CHECK_EQ(data.dim(), 4);
    CHECK_EQ(result.dim(), 4);
    CHECK_EQ(data.device(), result.device());
    CHECK_EQ(data.scalar_type(), result.scalar_type());
    CHECK_GE(std::min({pad_left, pad_right, pad_top, pad_bottom}), 0);
    CHECK_EQ(result.size(0), data.size(0));
    CHECK_EQ(result.size(1), data.size(1));
    CHECK_EQ(result.size(2), data.size(2) + pad_top + pad_bottom);
    CHECK_EQ(result.size(3), data.size(3) + pad_left + pad_right);
    SELECT_DEVICE(result.device(), padding_2d_impl, data, result, pad_left, pad_right, pad_top, pad_bottom, mode);
}
Tensor padding_2d(Tensor data, int pad_left, int pad_right, int pad_top, int pad_bottom, PaddingMode mode)
{
    CHECK_EQ(data.dim(), 4);
    auto out_sizes = data.sizes();
    out_sizes[2] += pad_top + pad_bottom;
    out_sizes[3] += pad_left + pad_right;
    CHECK_LE(out_sizes[2], 2 * data.size(2));
    CHECK_LE(out_sizes[3], 2 * data.size(3));
    auto result = empty(out_sizes, data.options());
    padding_2d(data, result, pad_left, pad_right, pad_top, pad_bottom, mode);
    return result;
}

void padding_3d(Tensor data, Tensor result, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front,
                int pad_back, PaddingMode mode)
{
    CHECK_EQ(data.dim(), 5);
    CHECK_EQ(result.dim(), 5);
    CHECK_EQ(data.device(), result.device());
    CHECK_EQ(data.scalar_type(), result.scalar_type());
    CHECK_GE(std::min({pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back}), 0);
    CHECK_EQ(result.size(0), data.size(0));
    CHECK_EQ(result.size(1), data.size(1));
    CHECK_EQ(result.size(2), data.size(2) + pad_front + pad_back);
    CHECK_EQ(result.size(3), data.size(3) + pad_top + pad_bottom);
    CHECK_EQ(result.size(4), data.size(4) + pad_left + pad_right);
    SELECT_DEVICE(result.device(), padding_3d_impl, data, result, pad_left, pad_right, pad_top, pad_bottom, pad_front,
                  pad_back, mode);
}

Tensor padding_3d(Tensor data, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front, int pad_back,
                  PaddingMode mode)
{
    CHECK_EQ(data.dim(), 5);  // 5D Tensor: [Batch, Channel, Depth, Height, Width]

    auto out_sizes = data.sizes();

    out_sizes[2] += pad_front + pad_back;
    out_sizes[3] += pad_top + pad_bottom;
    out_sizes[4] += pad_left + pad_right;

    CHECK_LE(out_sizes[2], 2 * data.size(2));
    CHECK_LE(out_sizes[3], 2 * data.size(3));
    CHECK_LE(out_sizes[4], 2 * data.size(4));

    auto result = empty(out_sizes, data.options());

    padding_3d(data, result, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back, mode);

    return result;
}

namespace autograd
{
struct SliceNode : public FunctionNode<SliceNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue dim, IValue start, IValue end, IValue step)
    {
        ctx->saved_data["dim"]   = dim;
        ctx->saved_data["start"] = start;
        ctx->saved_data["end"]   = end;
        ctx->saved_data["step"]  = step;
        ctx->save_for_backward({a});
        Tensor result = a.slice_view(dim.toInt(), start.toInt(), end.toInt(), step.toInt());
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int64_t dim   = ctx->saved_data["dim"].toInt();
        int64_t start = ctx->saved_data["start"].toInt();
        int64_t end   = ctx->saved_data["end"].toInt();
        int64_t step  = ctx->saved_data["step"].toInt();

        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto g      = grad[0];
        auto grad_a = zeros_like(a);

        // grad_a.slice_view(dim, start, end, step) += g;
        grad_a.slice_view(dim, start, end, step).copy_(g);

        return {grad_a, {}, {}, {}, {}};
    }
};
}  // namespace autograd
Tensor slice(Tensor a, int64_t dim, int64_t start, int64_t end, int64_t step)
{
    CHECK_GT(step, 0) << "slice step must be positive";
    dim = maybe_wrap_dim(dim, a.dim());
    return autograd::SliceNode::apply(a, dim, start, end, step)[0];
}

Tensor stack(const std::vector<Tensor>& tensors)
{
    std::vector<Tensor> tensors_unsqueeze;
    for (auto& t : tensors)
    {
        tensors_unsqueeze.push_back(t.unsqueeze(0));
    }
    return cat(tensors_unsqueeze, 0);
}


namespace autograd
{
struct CloneNode : public FunctionNode<CloneNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        Tensor result = empty_like(a);
        SELECT_DEVICE(result.device(), copy_and_convert_impl, a, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto g = grad[0];

        Tensor grad_a = empty_like(g);
        SELECT_DEVICE(grad_a.device(), copy_and_convert_impl, g, grad_a);
        return {grad_a};
    }
};


struct PermuteNode : public FunctionNode<PermuteNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue new_indices)
    {
        auto new_sizes       = a.sizes();
        auto reverse_indices = new_indices.toSizes();
        for (int i = 0; i < a.dim(); ++i)
        {
            reverse_indices[new_indices.toSizes()[i]] = i;
        }
        ctx->saved_data["reverse_indices"] = reverse_indices;
        Tensor result                      = a.permute_view(new_indices.toSizes());
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto g      = grad[0];
        auto grad_a = g.permute_view(ctx->saved_data["reverse_indices"].toSizes());
        return {grad_a, {}};
    }
};
}  // namespace autograd

Tensor clone(Tensor a)
{
    if (!a.defined()) return Tensor();
    return autograd::CloneNode::apply(a)[0];
}
Tensor permute(Tensor t, const SizeType& size)
{
    CHECK_EQ(size.size(), t.dim());
    SizeType dims = size;
    for (auto& d : dims.vec())
    {
        d = maybe_wrap_dim(d, t.dim());
    }
    return autograd::PermuteNode::apply(t, dims)[0];
}


Tensor flip(Tensor t, const SizeType& size)
{
    CHECK(!t.requires_grad());
    for (auto d : size.vec())
    {
        t = t.slice_view(d, t.size(d) - 1, -1, -1, false).clone();
    }
    return t;
}



namespace autograd
{
struct Cat2Node : public FunctionNode<Cat2Node>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b, IValue dim)
    {
        CHECK_EQ(a.dim(), b.dim()) << "cat: tensors must have the same number of dims";
        CHECK_EQ(a.device(), b.device()) << "cat: tensors must be on the same device";
        CHECK_EQ(a.scalar_type(), b.scalar_type()) << "cat: tensors must have the same dtype";
        for (int64_t d = 0; d < a.dim(); ++d)
        {
            CHECK(d == dim.toInt() || a.size(d) == b.size(d))
                << "cat: sizes must match except in dim " << dim.toInt() << ": " << a.sizes() << " " << b.sizes();
        }
        ctx->saved_data["dim"] = dim;
        auto output_size       = a.sizes();
        output_size[dim.toInt()] += b.size(dim.toInt());

        auto result = empty(output_size, a.options());
        result.slice_view(dim.toInt(), 0, 0 + a.size(dim.toInt())).copy_(a);
        result.slice_view(dim.toInt(), a.size(dim.toInt()), a.size(dim.toInt()) + b.size(dim.toInt())).copy_(b);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int64_t dim = ctx->saved_data["dim"].toInt();
        auto l      = ctx->get_saved_variables();
        // auto a      = l[0];
        // auto b      = l[1];

        auto a_size = ctx->next_meta[0].size;
        auto b_size = ctx->next_meta[1].size;
        auto g      = grad[0];

#if 0
        auto grad_a = empty_like(a);
        auto grad_b = empty_like(b);
        grad_a.copy_(g.slice_view(dim, 0, 0 + a.size(dim)));
        grad_b.copy_(g.slice_view(dim, a.size(dim), a.size(dim) + b.size(dim)));
#else
        auto grad_a = g.slice_view(dim, 0, 0 + a_size[dim]);
        auto grad_b = g.slice_view(dim, a_size[dim], a_size[dim] + b_size[dim]);

#endif

        return {grad_a, grad_b, {}};
    }
};
struct GridsampleNode : public FunctionNode<GridsampleNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor input, Tensor grid, IValue _interpolation, IValue _padding,
                                       IValue _align_corners)
    {
        CHECK(input.dim() == 4 || input.dim() == 5) << "grid_sample: input must be 4D or 5D, got " << input.dim();
        CHECK_EQ(grid.dim(), input.dim()) << "grid_sample: grid and input must have the same number of dims";
        CHECK_EQ(grid.size(0), input.size(0)) << "grid_sample: grid and input must have the same batch size";
        CHECK_EQ(grid.size(grid.dim() - 1), input.dim() - 2)
            << "grid_sample: the last grid dim must have size " << input.dim() - 2;
        CHECK_EQ(grid.device(), input.device()) << "grid_sample: grid and input must be on the same device";
        CHECK_EQ(grid.scalar_type(), input.scalar_type()) << "grid_sample: grid and input must have the same dtype";
        CHECK(input.scalar_type() == kHalf || input.scalar_type() == kFloat || input.scalar_type() == kDouble)
            << "grid_sample: only half, float and double are supported, got " << input.scalar_type();

        InterpolationType interpolation = (InterpolationType)_interpolation.toInt();
        PaddingMode padding             = (PaddingMode)_padding.toInt();
        CHECK(interpolation == kBilinear || interpolation == kNearest) << "grid_sample: invalid interpolation mode";
        CHECK(padding == kBorder || padding == kZero || padding == kReflect) << "grid_sample: invalid padding mode";
        bool align_corners = _align_corners.toBool();
        auto size_out      = input.sizes();
        size_out[2]        = grid.size(1);
        size_out[3]        = grid.size(2);
        if (input.dim() == 5)
        {
            size_out[4] = grid.size(3);
        }
        auto result = empty(size_out, input.options());


        if (input.dim() == 4)
        {
            SELECT_DEVICE(input.device(), grid_sample_2d_impl, input, grid, interpolation, padding, align_corners,
                          result);
        }
        else
        {
            SELECT_DEVICE(input.device(), grid_sample_3d_impl, input, grid, interpolation, padding, align_corners,
                          result);
        }
        ctx->saved_data["interpolation"] = _interpolation;
        ctx->saved_data["padding"]       = _padding;
        ctx->saved_data["align_corners"] = _align_corners;
        ctx->save_for_backward({input, grid});

        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l          = ctx->get_saved_variables();
        auto input      = l[0];
        auto grid       = l[1];
        auto g          = grad[0];
        auto grad_input = zeros_like(input);
        auto grad_grid  = zeros_like(grid);

        InterpolationType interpolation = (InterpolationType)ctx->saved_data["interpolation"].toInt();
        PaddingMode padding             = (PaddingMode)ctx->saved_data["padding"].toInt();
        bool align_corners              = ctx->saved_data["align_corners"].toBool();

        if (input.dim() == 4)
        {
            SELECT_DEVICE(input.device(), grid_sample_2d_backward_impl, input, grid, interpolation, padding,
                          align_corners, grad_input, grad_grid, g);
        }
        else
        {
            SELECT_DEVICE(input.device(), grid_sample_3d_backward_impl, input, grid, interpolation, padding,
                          align_corners, grad_input, grad_grid, g);
        }

        return {grad_input, grad_grid, {}, {}, {}};
    }
};
}  // namespace autograd

Tensor cat(const std::vector<Tensor>& list, int64_t dim)
{
    if (list.empty())
    {
        return {};
    }
    // like PyTorch, skip legacy empty 1-D tensors ({0}) when they are mixed with other tensors
    std::vector<Tensor> inputs;
    for (auto& t : list)
    {
        if (!(t.dim() == 1 && t.numel() == 0))
        {
            inputs.push_back(t);
        }
    }
    if (inputs.empty())
    {
        inputs = list;
    }

    auto result = inputs.front();
    dim         = maybe_wrap_dim(dim, result.dim());
    for (int i = 1; i < inputs.size(); ++i)
    {
        result = autograd::Cat2Node::apply(result, inputs[i], dim)[0];
    }

    return result;
}

std::pair<Tensor, Tensor> sort(Tensor t, int64_t dim)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    dim = maybe_wrap_dim(dim, t.dim());

    auto initial_device = t.device();
    t                   = t.cpu();

    auto result_t     = empty_like(t);
    auto result_index = empty(t.sizes(), TensorOptions().dtype<int64_t>());
    cpu_impl::sort_impl(t, dim, result_t, result_index);

    result_t     = result_t.to(initial_device);
    result_index = result_index.to(initial_device);


    return {result_t, result_index};
}
namespace autograd
{
struct Conv2dNode : public FunctionNode<Conv2dNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor input, Tensor weight, Tensor bias, IValue _stride,
                                       IValue _padding, IValue _dilation, IValue _groups)
    {
        int stride   = (int)_stride.toInt();
        int padding  = (int)_padding.toInt();
        int dilation = (int)_dilation.toInt();
        int groups   = (int)_groups.toInt();

        CHECK_EQ(input.dim(), 4) << "conv2d: input must be [N, Cin, H, W]";
        CHECK_EQ(weight.dim(), 4) << "conv2d: weight must be [Cout, Cin / groups, kH, kW]";
        CHECK_EQ(input.device(), weight.device());
        CHECK_EQ(input.scalar_type(), weight.scalar_type());
        CHECK_GE(stride, 1);
        CHECK_GE(padding, 0);
        CHECK_GE(dilation, 1);
        CHECK_GE(groups, 1);
        CHECK_EQ(input.size(1) % groups, 0);
        CHECK_EQ(weight.size(0) % groups, 0);
        CHECK_EQ(weight.size(1) * groups, input.size(1)) << "conv2d: weight and input channels do not match";
        if (bias.defined())
        {
            CHECK_EQ(bias.dim(), 1);
            CHECK_EQ(bias.size(0), weight.size(0));
            CHECK_EQ(bias.device(), input.device());
            CHECK_EQ(bias.scalar_type(), input.scalar_type());
        }

        int64_t out_height = (input.size(2) + 2 * padding - dilation * (weight.size(2) - 1) - 1) / stride + 1;
        int64_t out_width  = (input.size(3) + 2 * padding - dilation * (weight.size(3) - 1) - 1) / stride + 1;
        CHECK_GT(out_height, 0) << "conv2d: kernel larger than the padded input";
        CHECK_GT(out_width, 0) << "conv2d: kernel larger than the padded input";

        auto result = empty({input.size(0), weight.size(0), out_height, out_width}, input.options());
        SELECT_DEVICE(input.device(), conv2d, input, weight, bias, stride, padding, dilation, groups, result);

        ctx->saved_data["stride"]   = _stride;
        ctx->saved_data["padding"]  = _padding;
        ctx->saved_data["dilation"] = _dilation;
        ctx->saved_data["groups"]   = _groups;
        ctx->save_for_backward({input, weight});
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l       = ctx->get_saved_variables();
        auto input   = l[0];
        auto weight  = l[1];
        auto g       = grad[0];
        int stride   = (int)ctx->saved_data["stride"].toInt();
        int padding  = (int)ctx->saved_data["padding"].toInt();
        int dilation = (int)ctx->saved_data["dilation"].toInt();
        int groups   = (int)ctx->saved_data["groups"].toInt();

        Tensor grad_input, grad_weight, grad_bias;
        if (ctx->requires_grad_for_input(0))
        {
            grad_input = empty(input.sizes(), input.options());
            SELECT_DEVICE(input.device(), conv2d_backward_input, weight, g, stride, padding, dilation, groups,
                          grad_input);
        }
        if (ctx->requires_grad_for_input(1))
        {
            grad_weight = empty(weight.sizes(), weight.options());
            SELECT_DEVICE(input.device(), conv2d_backward_weight, input, g, stride, padding, dilation, groups,
                          grad_weight);
        }
        if (ctx->requires_grad_for_input(2))
        {
            grad_bias = g.sum({0, 2, 3}, false);
        }
        return {grad_input, grad_weight, grad_bias, {}, {}, {}, {}};
    }
};
}  // namespace autograd

Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups)
{
    return autograd::Conv2dNode::apply(input, weight, bias, (int64_t)stride, (int64_t)padding, (int64_t)dilation,
                                       (int64_t)groups)[0];
}

Tensor nn::functional::grid_sample(Tensor data, Tensor uv, nn::functional::GridSampleFuncOptions options)
{
    return autograd::GridsampleNode::apply(data, uv, options.it, options.pm, options.ac)[0];
}

namespace autograd
{
// Batched matmul of a [..., M, K] and b [..., K, N] with broadcast batch dimensions (both operands >= 2-D).
static Tensor matmul_broadcast(Tensor a, Tensor b)
{
    int64_t a_dim = a.dim();
    int64_t b_dim = b.dim();
    CHECK_GE(a_dim, 2);
    CHECK_GE(b_dim, 2);
    CHECK_EQ(a.device(), b.device());
    CHECK_EQ(a.scalar_type(), b.scalar_type());
    CHECK_EQ(a.size(a_dim - 1), b.size(b_dim - 2)) << "matmul: inner dimensions do not match";

    int64_t batch_dim = std::max(a_dim, b_dim) - 2;
    std::vector<int64_t> out_sizes(batch_dim + 2);
    for (int64_t d = 0; d < batch_dim; ++d)
    {
        int64_t da = d - (batch_dim - (a_dim - 2));
        int64_t db = d - (batch_dim - (b_dim - 2));
        int64_t sa = da >= 0 ? a.size(da) : 1;
        int64_t sb = db >= 0 ? b.size(db) : 1;
        CHECK(sa == sb || sa == 1 || sb == 1) << "matmul: batch dimensions are not broadcastable";
        out_sizes[d] = sa == 1 ? sb : sa;
    }
    out_sizes[batch_dim]     = a.size(a_dim - 2);
    out_sizes[batch_dim + 1] = b.size(b_dim - 1);

    Tensor result = empty(out_sizes, a.options());
    if (result.numel() > 0)
    {
        SELECT_DEVICE(a.device(), matmul_impl, a, b, result);
    }
    return result;
}

// Sums the gradient of a broadcast operand back to the operand's sizes.
static Tensor reduce_to_sizes(Tensor g, const SizeType& sizes)
{
    int64_t extra = g.dim() - (int64_t)sizes.size();
    CHECK_GE(extra, 0);
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < g.dim(); ++d)
    {
        if (d < extra || (sizes[d - extra] == 1 && g.size(d) != 1))
        {
            reduce_dims.push_back(d);
        }
    }
    if (!reduce_dims.empty())
    {
        g = g.sum(reduce_dims, true);
    }
    if (extra > 0)
    {
        g = g.reshape(sizes);
    }
    return g;
}

struct MatmulNode : public FunctionNode<MatmulNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b)
    {
        ctx->save_for_backward({a, b});
        return {matmul_broadcast(a, b)};
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
            // dL/dA = dL/dC * B^T
            grad_a = reduce_to_sizes(matmul_broadcast(g, transpose(b, b.dim() - 2, b.dim() - 1)), a.sizes());
        }
        if (ctx->requires_grad_for_input(1))
        {
            // dL/dB = A^T * dL/dC
            grad_b = reduce_to_sizes(matmul_broadcast(transpose(a, a.dim() - 2, a.dim() - 1), g), b.sizes());
        }
        return {grad_a, grad_b};
    }
};
}  // namespace autograd

Tensor matmul(Tensor a, Tensor b)
{
    CHECK(a.dim() >= 1 && b.dim() >= 1) << "matmul: both arguments need at least one dimension";
    if (a.dim() == 1 && b.dim() == 1)
    {
        return autograd::MatmulNode::apply(a.reshape({1, a.size(0)}), b.reshape({b.size(0), 1}))[0].reshape({1});
    }
    if (a.dim() == 1)
    {
        // [K] @ [..., K, N] -> [..., N]
        auto r                     = autograd::MatmulNode::apply(a.reshape({1, a.size(0)}), b)[0];
        std::vector<int64_t> sizes = r.sizes().vec();
        sizes.erase(sizes.end() - 2);
        return r.reshape(sizes);
    }
    if (b.dim() == 1)
    {
        // [..., M, K] @ [K] -> [..., M]
        auto r                     = autograd::MatmulNode::apply(a, b.reshape({b.size(0), 1}))[0];
        std::vector<int64_t> sizes = r.sizes().vec();
        sizes.pop_back();
        return r.reshape(sizes);
    }
    return autograd::MatmulNode::apply(a, b)[0];
}
}  // namespace tinytorch
