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
        auto g = grad[0];
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


Tensor index_select(Tensor input, int64_t dim, Tensor index)
{
    CHECK(!input.requires_grad() || !GradMode::is_enabled());

    
    CHECK_LT(dim, input.dim());
    CHECK_EQ(index.dtype(), kInt32);

    SizeType result_size = input.sizes();
    result_size[dim]     = index.numel();

    Tensor result = empty(result_size, input.options());
    if (input.is_cpu())
    {
        index_select_impl_cpu(input, dim, index, result);
    }
    else
    {
        index_select_impl_cuda(input, dim, index, result);
    }
    return result;
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
        CHECK_EQ(index.dtype(), kInt32);
        CHECK_EQ(input.dim(), data.dim());
        CHECK_EQ(index.dim(), 1);
        CHECK_EQ(index.numel(), data.size(0));

        Tensor result = input.clone();
        index_add_impl_cpu(input, dim.toInt(), index, data, result);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l     = ctx->get_saved_variables();
        auto index = l[0];

        auto grad_data  = index_select(grad[0], ctx->saved_data["dim"].toInt(), index);
        auto grad_input = grad[0];

        // Tensor grad_a = empty_like(grad[0]);
        // copy_impl_cpu(grad[0], grad_a);
        return {grad_input, {}, {}, grad_data};
    }
};
}  // namespace autograd

Tensor index_add(Tensor input, int64_t dim, Tensor index, Tensor data)
{
    // CHECK(!input.requires_grad() || !GradMode::is_enabled());
    // CHECK(!data.requires_grad() || !GradMode::is_enabled());

    return autograd::IndexAddNode::apply(input, dim, index, data)[0];
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

Tensor cat(const std::vector<Tensor>& list, int64_t dim)
{
    CHECK_GT(list.size(), 0);
    auto output_size        = list.front().sizes();
    int64_t target_dim_size = 0;
    for (auto a : list)
    {
        target_dim_size += a.size(dim);
    }
    output_size[dim] = target_dim_size;

    auto result = empty(output_size, list.front().options());

    int64_t current_offset = 0;
    for (auto a : list)
    {
        result.slice(dim, current_offset, current_offset + a.size(dim)).copy_(a);
    }

    return result;
}

}  // namespace tinytorch