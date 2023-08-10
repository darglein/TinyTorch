/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl.h"


namespace tinytorch
{

static void fill_neg_one_dim(SizeType& new_sizes, int64_t old_numel)
{
    int64_t new_numel = 1;
    int64_t* neg_dim  = nullptr;
    for (int64_t& i : new_sizes.vec())
    {
        if (i == -1)
        {
            CHECK_EQ(neg_dim, nullptr);
            neg_dim = &i;
        }
        else
        {
            new_numel *= i;
        }
    }

    if (neg_dim)
    {
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

Tensor repeat(Tensor t, SizeType sizes)
{
    CHECK_EQ(t.dim(), sizes.size());

    int repeat_dim = -1;
    for (int i = 0; i < sizes.size(); ++i)
    {
        if (sizes[i] > 1)
        {
            CHECK_EQ(repeat_dim, -1);
            repeat_dim = i;
        }
    }

    if (repeat_dim == -1)
    {
        return t.clone();
    }



    auto result = t;


    for (int i = 0; i < sizes[repeat_dim] - 1; ++i)
    {
        result = cat({result, t}, repeat_dim);
    }

    //     std::cout << "new_size " << new_size << std::endl;
    // throw std::runtime_error("lsdf");
    return result;
}

Tensor repeat_interleave(Tensor t, int64_t count)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());

    SizeType new_sizes = t.sizes();
    new_sizes[0] *= count;
    Tensor result = empty(new_sizes, t.options());

    SELECT_DEVICE(t.device(), repeat_interleave_impl, t, count, result);
    return result;
}

Tensor transpose(Tensor t, int64_t dim0, int64_t dim1)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());

    SizeType new_sizes = t.sizes();
    std::swap(new_sizes[dim0], new_sizes[dim1]);

    Tensor result = empty(new_sizes, t.options());
    SELECT_DEVICE(t.device(), transpose_impl, t, dim0, dim1, result);
    return result;
}

void fill(Tensor& t, double value)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(t.device(), fill_impl, t, value);
}

void fill(Tensor& t, Tensor value)
{
    CHECK_EQ(value.numel(), 1);
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(t.device(), fill_impl, t, value);
}
void fill(Tensor& t, Tensor values, int dim)
{
    CHECK_EQ(values.numel(), t.size(dim));
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(t.device(), fill_impl, t, values, dim);
}

void uniform(Tensor& t, double mi, double ma)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(t.device(), uniform_impl, t, mi, ma);
}

void uniform_int(Tensor& t, int low, int high)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(t.device(), uniform_int_impl, t, low, high);
}

void copy(Tensor src, Tensor target)
{
    CHECK(!src.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(src.device(), copy_and_convert_impl, src, target);
}



namespace autograd
{
struct ToNode : public FunctionNode<ToNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue new_device_)
    {
        NoGradGuard ngg;
        Device new_device = new_device_.toDevice();
#ifdef TT_HAS_CUDA
        Tensor contig = a.contiguous();
        Tensor result = empty(contig.sizes(), a.options().device(new_device));
        cpu_impl::to_impl_cpu_cuda(contig, result);
        return {result};
#else
        CHECK(false);
#endif
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        Device old_device = ctx->next_meta[0].device;
        Tensor grad_a     = empty_like(grad[0], grad[0].options().device(old_device));
        cpu_impl::to_impl_cpu_cuda(grad[0], grad_a);
        return {grad_a, {}};
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


Tensor to(Tensor a, Device new_device)
{
    if (a.device() == new_device)
    {
        return a;
    }

    return autograd::ToNode::apply(a.contiguous(), new_device)[0];
}

Tensor to(Tensor a, ScalarType other_type)
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
        ctx->saved_data["dim"] = dim;
        ctx->save_for_backward({index});


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
        int dim          = ctx->saved_data["dim"].toInt();
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
        CHECK_EQ(input.dim(), data.dim());
        CHECK_EQ(index.dim(), 1);
        CHECK_EQ(index.numel(), data.size(dim.toInt())) << index.sizes() << " " << data.sizes();

        Tensor result = input.clone();
        SELECT_DEVICE(result.device(), index_add_impl, dim.toInt(), index, data, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int dim    = ctx->saved_data["dim"].toInt();
        auto l     = ctx->get_saved_variables();
        auto index = l[0];

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
    return autograd::IndexAddNode::apply(input, dim, index, data)[0];
}

void index_copy_(Tensor& target, int64_t dim, Tensor index, Tensor value)
{
    CHECK(!target.requires_grad() || !GradMode::is_enabled());
    SELECT_DEVICE(target.device(), index_copy_impl, target, dim, index, value);
}

Tensor gather(Tensor data, int64_t dim, Tensor index)
{
    CHECK(!data.requires_grad() || !GradMode::is_enabled());
    CHECK_EQ(data.dim(), index.dim());
    auto out_sizes = index.sizes();
    auto result = empty(out_sizes, data.options());
    SELECT_DEVICE(result.device(), gather_impl, data, dim, index, result);
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
        int dim   = ctx->saved_data["dim"].toInt();
        int start = ctx->saved_data["start"].toInt();
        int end   = ctx->saved_data["end"].toInt();
        int step  = ctx->saved_data["step"].toInt();

        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto g      = grad[0];
        auto grad_a = zeros_like(a);

        grad_a.slice_view(dim, start, end, step) += g;

        return {grad_a, {}, {}, {}, {}};
    }
};
}  // namespace autograd
Tensor slice(Tensor a, int64_t dim, int64_t start, int64_t end, int64_t step)
{
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
        auto g = grad[0];

        auto grad_a = g.permute_view(ctx->saved_data["reverse_indices"].toSizes());
        // Tensor grad_a = empty(ctx->next_meta[0].size, g.options());
        // SELECT_DEVICE(grad_a.device(), permute_impl, g, grad_a, ctx->saved_data["reverse_indices"].toSizes());
        return {grad_a, {}};
    }
};
}  // namespace autograd

Tensor clone(Tensor a)
{
    return autograd::CloneNode::apply(a)[0];
}
Tensor permute(Tensor t, const SizeType& size)
{
    return autograd::PermuteNode::apply(t, size)[0];
}


namespace autograd
{
struct Cat2Node : public FunctionNode<Cat2Node>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b, IValue dim)
    {
        ctx->saved_data["dim"] = dim;
        auto output_size       = a.sizes();
        output_size[dim.toInt()] += b.size(dim.toInt());

        auto result = empty(output_size, a.options());
        result.slice_view(dim.toInt(), 0, 0 + a.size(dim.toInt())).copy_(a);
        result.slice_view(dim.toInt(), a.size(dim.toInt()), a.size(dim.toInt()) + b.size(dim.toInt())).copy_(b);

        ctx->save_for_backward({a, b});
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int dim     = ctx->saved_data["dim"].toInt();
        auto l      = ctx->get_saved_variables();
        auto a      = l[0];
        auto b      = l[1];
        auto g      = grad[0];
        auto grad_a = zeros_like(a);
        auto grad_b = zeros_like(b);

        grad_a.copy_(g.slice_view(dim, 0, 0 + a.size(dim)));
        grad_b.copy_(g.slice_view(dim, a.size(dim), a.size(dim) + b.size(dim)));

        return {grad_a, grad_b, {}};
    }
};
struct GridsampleNode : public FunctionNode<GridsampleNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor input, Tensor grid, IValue _interpolation, IValue _padding,
                                       IValue _align_corners)
    {
        CHECK(input.dim() == 4 || input.dim() == 5);

        InterpolationType interpolation = (InterpolationType)_interpolation.toInt();
        PaddingMode padding             = (PaddingMode)_padding.toInt();
        CHECK_EQ(interpolation, InterpolationType::kBilinear);
        CHECK_EQ(padding, PaddingMode::kBorder);
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
    auto result = list.front();
    for (int i = 1; i < list.size(); ++i)
    {
        result = autograd::Cat2Node::apply(result, list[i], dim)[0];
    }

    return result;
}

std::pair<Tensor, Tensor> sort(Tensor t, int64_t dim)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());

    auto initial_device = t.device();
    t                   = t.cpu();

    auto result_t     = empty_like(t);
    auto result_index = empty(t.sizes(), TensorOptions().dtype<int64_t>());
    cpu_impl::sort_impl(t, dim, result_t, result_index);

    result_t     = result_t.to(initial_device);
    result_index = result_index.to(initial_device);


    return {result_t, result_index};
}


Tensor nn::functional::grid_sample(Tensor data, Tensor uv, nn::functional::GridSampleFuncOptions options)
{
    return autograd::GridsampleNode::apply(data, uv, options.it, options.pm, options.ac)[0];
}

}  // namespace tinytorch
