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
    return repeat_interleave_impl_cpu(t, count);
}

Tensor transpose(Tensor t, int64_t dim0, int64_t dim1)
{
    CHECK(!t.requires_grad() || !GradMode::is_enabled());
    return transpose_impl_cpu(t, dim0, dim1);
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
        fill_impl_cuda(t, value);
    }
}

void copy(Tensor src, Tensor target)
{
    CHECK(!src.requires_grad() || !GradMode::is_enabled());
    copy_impl_cpu(src, target);
}

Tensor to(Tensor a, ScalarType other_type)
{
    CHECK(!a.requires_grad() || !GradMode::is_enabled());
    return to_impl_cpu(a, other_type);
}


namespace autograd
{
struct ToNode : public FunctionNode<ToNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor a, IValue new_device_)
    {
        NoGradGuard ngg;
        // ctx->save_for_backward({a});
        // Tensor result = empty_like(a);
        // copy_impl_cpu(a, result);
        // return {result};

        ctx->saved_data["old_device"] = (int)a.device();

        Device new_device = (Device)new_device_.toInt();

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


        // auto l        = ctx->get_saved_variables();
        Tensor grad_a = empty_like(grad[0], grad[0].options().device(old_device));
        to_impl_cpu_cuda(grad[0], grad_a);
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


Tensor index_select(Tensor input, int64_t dim, Tensor index)
{
    CHECK(!input.requires_grad() || !GradMode::is_enabled());
    return index_select_impl_cpu(input, dim, index);
}

namespace autograd
{
struct IndexAddNode : public FunctionNode<IndexAddNode>
{
    static std::vector<Tensor> forward(Context* ctx, Tensor input, IValue dim, Tensor index, Tensor data)
    {
        ctx->saved_data["dim"] = dim;
        ctx->save_for_backward({index});
        auto result = index_add_impl_cpu(input, dim.toInt(), index, data);
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        auto l     = ctx->get_saved_variables();
        auto index = l[0];

        auto grad_data  = index_select_impl_cpu(grad[0], ctx->saved_data["dim"].toInt(), index);
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

Tensor stack(const std::vector<Tensor>& a)
{
    for (auto b : a)
    {
        CHECK(!b.requires_grad() || !GradMode::is_enabled());
    }
    return stack_impl_cpu(a);
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
            copy_impl_cpu(a, result);
        }
        else
        {
            copy_impl_cuda(a, result);
        }
        return {result};
    }

    static std::vector<Tensor> backward(Context* ctx, const std::vector<Tensor>& grad)
    {
        // auto l        = ctx->get_saved_variables();
        Tensor grad_a = empty_like(grad[0]);
        if (grad_a.is_cpu())
        {
            copy_impl_cpu(grad[0], grad_a);
        }
        else
        {
            copy_impl_cuda(grad[0], grad_a);
        }
        return {grad_a[0]};
    }
};
}  // namespace autograd

Tensor clone(Tensor a)
{
    return autograd::CloneNode::apply(a)[0];
}

}  // namespace tinytorch