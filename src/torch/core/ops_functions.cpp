/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops_functions.h"

#include "graph.h"

#include "torch/cpu/ops_impl_cpu.h"
#include "torch/cuda/ops_impl_cuda.h"


namespace tinytorch
{

Tensor repeat(Tensor t, SizeType sizes)
{
    CHECK(!t.requires_grad());
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


    auto new_size        = t.sizes().vec();
    new_size[repeat_dim] = new_size[repeat_dim] * sizes[repeat_dim];


    Tensor result = empty(new_size, t.options());

    for (int i = 0; i < sizes[repeat_dim]; ++i)
    {
        result.slice(repeat_dim, i * t.size(repeat_dim), (i + 1) * t.size(repeat_dim)).copy_(t);
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
    repeat_interleave_impl_cpu(t, count, result);
    return result;
}

Tensor transpose(Tensor t, int64_t dim0, int64_t dim1)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());

    SizeType new_sizes = t.sizes();
    std::swap(new_sizes[dim0], new_sizes[dim1]);

    Tensor result = empty(new_sizes, t.options());

    transpose_impl_cpu(t, dim0, dim1, result);
    return result;
}

void fill(Tensor& t, double value)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    if (t.is_cpu())
    {
        fill_impl_cpu(t, value);
    }
    else
    {
#ifdef TT_HAS_CUDA
        fill_impl_cuda(t, value);
#endif
    }
}

void uniform(Tensor& t, double mi, double ma)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    if (t.is_cpu())
    {
        uniform_impl_cpu(t, mi, ma);
    }
    else
    {
#ifdef TT_HAS_CUDA
        uniform_impl_cuda(t, mi, ma);
#endif
    }
}

void uniform_int(Tensor& t, int low, int high)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    if (t.is_cpu())
    {
        uniform_int_impl_cpu(t, low, high);
    }
    else
    {
#ifdef TT_HAS_CUDA
        uniform_int_impl_cuda(t, low, high);
#endif
    }
}

void copy(Tensor src, Tensor target)
{
    CHECK(!src.requires_grad() || !GradMode::is_enabled());

    if (src.is_cpu())
    {
        copy_and_convert_impl_cpu(src, target);
    }
    else
    {
        copy_and_convert_impl_cuda(src, target);
    }
}



namespace autograd
{
struct ToNode : public FunctionNode<ToNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue new_device_)
    {
        NoGradGuard ngg;
        ctx->saved_data["old_device"] = (int)a.device();
        Device new_device             = (Device)new_device_.toInt();

#ifdef TT_HAS_CUDA
        Tensor contig = a.contiguous();
        Tensor result = empty(contig.sizes(), a.options().device(new_device));
        to_impl_cpu_cuda(a, result);
        return {result};
#else
        CHECK(false);
#endif
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        Device old_device = (Device)ctx->saved_data["old_device"].toInt();
        Tensor grad_a     = empty_like(grad[0], grad[0].options().device(old_device));
        to_impl_cpu_cuda(grad[0], grad_a);
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
        if (a.is_cpu())
        {
            copy_and_convert_impl_cpu(a, result);
        }
        else
        {
            copy_and_convert_impl_cuda(a, result);
        }
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        Dtype old_dtype = (Dtype)ctx->saved_data["old_dtype"].toInt();
        auto g          = grad[0];
        Tensor grad_a   = empty_like(grad[0], grad[0].options().dtype(old_dtype));

        copy_and_convert_impl_cpu(g, grad_a);
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

    return autograd::ToNode::apply(a, (int)new_device)[0];
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
        ctx->saved_data["dim"]        = dim;
        ctx->data_sizes["input_size"] = input.sizes();
        ctx->save_for_backward({index});


        CHECK_LT(dim.toInt(), input.dim());
        CHECK(index.dtype() == kInt32 || index.dtype() == kInt64);
        CHECK_EQ(index.dim(), 1);



        CHECK(!input.requires_grad() || !GradMode::is_enabled());

        auto result_size         = input.sizes().vec();
        result_size[dim.toInt()] = index.numel();
        Tensor result            = empty(result_size, input.options());
        if (result.is_cpu())
        {
            index_select_impl_cpu(input, dim.toInt(), index, result);
        }
        else
        {
            index_select_impl_cuda(input, dim.toInt(), index, result);
        }
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        int dim          = ctx->saved_data["dim"].toInt();
        auto l           = ctx->get_saved_variables();
        auto index       = l[0];
        auto input_sizes = ctx->data_sizes["input_size"];

        auto g = grad[0];

        auto grad_input = zeros(input_sizes, g.options());

        index_add(g, dim, index, grad_input);
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
        CHECK_EQ(index.numel(), data.size(0));

        Tensor result = input.clone();
        index_add_impl_cpu(input, dim.toInt(), index, data, result);
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
        // copy_impl_cpu(grad[0], grad_a);
        return {grad_input, {}, {}, grad_data};
    }
};
}  // namespace autograd

Tensor index_add(Tensor input, int64_t dim, Tensor index, Tensor data)
{
    return autograd::IndexAddNode::apply(input, dim, index, data)[0];
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
    for (auto b : tensors)
    {
        CHECK(!b.requires_grad() || !GradMode::is_enabled());
    }


    if (tensors.empty())
    {
        return {};
    }

    for (const auto& t : tensors)
    {
        CHECK_EQ(tensors.front().sizes(), t.sizes());
        CHECK_EQ(tensors.front().device(), t.device());
        CHECK_EQ(tensors.front().scalar_type(), t.scalar_type());
    }

    SizeType new_sizes = tensors.front().sizes();
    new_sizes.vec().insert(new_sizes.vec().begin(), tensors.size());

    Tensor result = empty(new_sizes, tensors.front().options());

    stack_impl_cpu(tensors, result);
    return result;
}


namespace autograd
{
struct CloneNode : public FunctionNode<CloneNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a)
    {
        // ctx->save_for_backward({a});
        Tensor result = empty_like(a);

        if (a.is_cpu())
        {
            copy_and_convert_impl_cpu(a, result);
        }
        else
        {
#ifdef TT_HAS_CUDA
            copy_and_convert_impl_cuda(a, result);
#endif
        }
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        // auto l        = ctx->get_saved_variables();
        Tensor grad_a = empty_like(grad[0]);
        if (grad_a.is_cpu())
        {
            copy_and_convert_impl_cpu(grad[0], grad_a);
        }
        else
        {
#ifdef TT_HAS_CUDA
            copy_and_convert_impl_cuda(grad[0], grad_a);
#endif
        }
        return {grad_a[0]};
    }
};
}  // namespace autograd

Tensor clone(Tensor a)
{
    return autograd::CloneNode::apply(a)[0];
}
Tensor permute(Tensor t, const SizeType& size)
{
    throw std::runtime_error("not implemented");
    return Tensor();
}


namespace autograd
{
struct Cat2Node : public FunctionNode<Cat2Node>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, Tensor b, IValue dim)
    {
        auto output_size = a.sizes();
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

}  // namespace tinytorch