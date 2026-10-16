/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "conv.h"

#include "torch/cuda/ops_impl_cuda_helper.h"
#include <type_traits>

namespace tinytorch
{
namespace cuda_impl
{

template <typename T>
using ConvAccType = std::conditional_t<std::is_same_v<T, half>, float, T>;

template <typename T>
__launch_bounds__(128) static __global__
    void conv2d_impl_kernel(TensorInfoCuda<T, 4> input, TensorInfoCuda<T, 4> weight, TensorInfoCuda<T, 1> bias,
                            int stride, int padding, int dilation, int groups, TensorInfoCuda<T, 4> result)
{
    using G   = ConvAccType<T>;
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t cin_per_group  = weight.size(1);
    int64_t cout_per_group = weight.size(0) / groups;

    auto idx   = result.LinearIndexToDimIndex(i);
    int64_t n  = idx[0];
    int64_t co = idx[1];
    int64_t oy = idx[2];
    int64_t ox = idx[3];

    int64_t ci_begin = co / cout_per_group * cin_per_group;

    G sum = bias.data != nullptr ? G(bias(co)) : G(0);
    for (int64_t ci = 0; ci < cin_per_group; ++ci)
    {
        for (int64_t ky = 0; ky < weight.size(2); ++ky)
        {
            int64_t iy = oy * stride - padding + ky * dilation;
            if (iy < 0 || iy >= input.size(2)) continue;
            for (int64_t kx = 0; kx < weight.size(3); ++kx)
            {
                int64_t ix = ox * stride - padding + kx * dilation;
                if (ix < 0 || ix >= input.size(3)) continue;
                sum += G(weight(co, ci, ky, kx)) * G(input(n, ci_begin + ci, iy, ix));
            }
        }
    }
    result(n, co, oy, ox) = T(sum);
}

template <typename T>
__launch_bounds__(128) static __global__
    void conv2d_backward_input_kernel(TensorInfoCuda<T, 4> weight, TensorInfoCuda<T, 4> grad_result, int stride,
                                      int padding, int dilation, int groups, TensorInfoCuda<T, 4> grad_input)
{
    using G   = ConvAccType<T>;
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= grad_input.numel()) return;

    int64_t cin_per_group  = weight.size(1);
    int64_t cout_per_group = weight.size(0) / groups;

    auto idx   = grad_input.LinearIndexToDimIndex(i);
    int64_t n  = idx[0];
    int64_t ci = idx[1];
    int64_t iy = idx[2];
    int64_t ix = idx[3];

    int64_t group    = ci / cin_per_group;
    int64_t ci_local = ci % cin_per_group;

    G sum = G(0);
    for (int64_t co = group * cout_per_group; co < (group + 1) * cout_per_group; ++co)
    {
        for (int64_t ky = 0; ky < weight.size(2); ++ky)
        {
            int64_t ty = iy + padding - ky * dilation;
            if (ty < 0 || ty % stride != 0 || ty / stride >= grad_result.size(2)) continue;
            for (int64_t kx = 0; kx < weight.size(3); ++kx)
            {
                int64_t tx = ix + padding - kx * dilation;
                if (tx < 0 || tx % stride != 0 || tx / stride >= grad_result.size(3)) continue;
                sum += G(weight(co, ci_local, ky, kx)) * G(grad_result(n, co, ty / stride, tx / stride));
            }
        }
    }
    grad_input(n, ci, iy, ix) = T(sum);
}

template <typename T>
__launch_bounds__(128) static __global__
    void conv2d_backward_weight_kernel(TensorInfoCuda<T, 4> input, TensorInfoCuda<T, 4> grad_result, int stride,
                                       int padding, int dilation, int groups, TensorInfoCuda<T, 4> grad_weight)
{
    using G   = ConvAccType<T>;
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= grad_weight.numel()) return;

    int64_t cin_per_group  = grad_weight.size(1);
    int64_t cout_per_group = grad_weight.size(0) / groups;

    auto idx   = grad_weight.LinearIndexToDimIndex(i);
    int64_t co = idx[0];
    int64_t cl = idx[1];
    int64_t ky = idx[2];
    int64_t kx = idx[3];

    int64_t ci = co / cout_per_group * cin_per_group + cl;

    G sum = G(0);
    for (int64_t n = 0; n < grad_result.size(0); ++n)
    {
        for (int64_t oy = 0; oy < grad_result.size(2); ++oy)
        {
            int64_t iy = oy * stride - padding + ky * dilation;
            if (iy < 0 || iy >= input.size(2)) continue;
            for (int64_t ox = 0; ox < grad_result.size(3); ++ox)
            {
                int64_t ix = ox * stride - padding + kx * dilation;
                if (ix < 0 || ix >= input.size(3)) continue;
                sum += G(grad_result(n, co, oy, ox)) * G(input(n, ci, iy, ix));
            }
        }
    }
    grad_weight(co, cl, ky, kx) = T(sum);
}

void conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups, Tensor result)
{
    CUDA_SWITCH_MACRO_FLOAT(input.device(), input.scalar_type(), result.numel(), conv2d_impl_kernel, input, weight,
                            bias, stride, padding, dilation, groups, result);
}

void conv2d_backward_input(Tensor weight, Tensor grad_result, int stride, int padding, int dilation, int groups,
                           Tensor grad_input)
{
    CUDA_SWITCH_MACRO_FLOAT(weight.device(), weight.scalar_type(), grad_input.numel(), conv2d_backward_input_kernel,
                            weight, grad_result, stride, padding, dilation, groups, grad_input);
}

void conv2d_backward_weight(Tensor input, Tensor grad_result, int stride, int padding, int dilation, int groups,
                            Tensor grad_weight)
{
    CUDA_SWITCH_MACRO_FLOAT(input.device(), input.scalar_type(), grad_weight.numel(), conv2d_backward_weight_kernel,
                            input, grad_result, stride, padding, dilation, groups, grad_weight);
}
}  // namespace cuda_impl
}  // namespace tinytorch
