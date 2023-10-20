/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops/ops_impl.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include "unary_operators.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{

namespace cuda_impl
{

template <typename T, typename Op>
__launch_bounds__(128) static __global__
    void unary_operator_kernel(Op op, TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    using G   = typename CpuComputeFloatType<T>::Type;
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i < result.numel())
    {
        G input   = G(a[i]);
        G output  = op.forward(input);
        result[i] = T(output);
    }
}

template <typename T, typename Op>
__launch_bounds__(128) static __global__
    void unary_operator_backward_kernel(Op op, TensorInfoCuda<T> a, TensorInfoCuda<T> grad_a,
                                        TensorInfoCuda<T> grad_result)
{
    using G   = typename CpuComputeFloatType<T>::Type;
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= grad_a.numel()) return;
    {
        G input       = G(a[i]);
        G grad_output = G(grad_result[i]);
        G grad_input  = op.backward(input, grad_output);
        grad_a[i]     = T(grad_input);
    }
}


template<typename Op>
static void unary_operator_forward_helper(Op op, Tensor input, Tensor result)
{
    switch (input.scalar_type())                                                                    \
    {                                                                                               \
        CUDA_CASE_MACRO((unary_operator_kernel<uint8_t>), kUInt8, input.numel(), op, input, result) \
        CUDA_CASE_MACRO((unary_operator_kernel<int16_t>), kInt16, input.numel(), op, input, result) \
        CUDA_CASE_MACRO((unary_operator_kernel<int32_t>), kInt32, input.numel(), op, input, result) \
        CUDA_CASE_MACRO((unary_operator_kernel<int64_t>), kLong, input.numel(), op, input, result)  \
        CUDA_CASE_MACRO((unary_operator_kernel<__half>), kHalf, input.numel(), op, input, result)   \
        CUDA_CASE_MACRO((unary_operator_kernel<float>), kFloat, input.numel(), op, input, result)   \
        CUDA_CASE_MACRO((unary_operator_kernel<double>), kDouble, input.numel(), op, input, result) \
        default:                                                                                    \
            CHECK(false) << "invalid input type " << input.scalar_type();                           \
    }
}
template<typename Op>
static void unary_operator_backward_helper(Op op, Tensor input, Tensor grad_input, Tensor grad_result)
{
    switch (input.scalar_type())                                                                                 \
    {                                                                                                            \
        CUDA_CASE_MACRO((unary_operator_backward_kernel<uint8_t>), kUInt8, input.numel(), op, input, grad_input, \
                        grad_result)                                                                             \
        CUDA_CASE_MACRO((unary_operator_backward_kernel<int16_t>), kInt16, input.numel(), op, input, grad_input, \
                        grad_result)                                                                             \
        CUDA_CASE_MACRO((unary_operator_backward_kernel<int32_t>), kInt32, input.numel(), op, input, grad_input, \
                        grad_result)                                                                             \
        CUDA_CASE_MACRO((unary_operator_backward_kernel<int64_t>), kLong, input.numel(), op, input, grad_input,  \
                        grad_result)                                                                             \
        CUDA_CASE_MACRO((unary_operator_backward_kernel<__half>), kHalf, input.numel(), op, input, grad_input,   \
                        grad_result)                                                                             \
        CUDA_CASE_MACRO((unary_operator_backward_kernel<float>), kFloat, input.numel(), op, input, grad_input,   \
                        grad_result)                                                                             \
        CUDA_CASE_MACRO((unary_operator_backward_kernel<double>), kDouble, input.numel(), op, input, grad_input, \
                        grad_result)                                                                             \
        default:                                                                                                 \
            CHECK(false) << "invalid input type " << input.scalar_type();                                        \
    }
}



void abs_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Abs(), a, result);
}
void round_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Round(), a, result);
}
void sqrt_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Sqrt(), a, result);
}
void sqrt_backward_impl(Tensor a, Tensor grad_a, Tensor grad_result)
{
    unary_operator_backward_helper(UnaryOperators::Sqrt(), a, grad_a, grad_result);
}

void log_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Log(), a, result);
}
void exp_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Exp(), a, result);
}
void sign_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Sign(), a, result);
}
void sin_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Sin(), a, result);
}
void cos_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Cos(), a, result);
}
void relu_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Relu(), a, result);
}
void sigmoid_impl(Tensor a, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Sigmoid(), a, result);
}
void sigmoid_backward_impl(Tensor a, Tensor grad_a, Tensor grad_result)
{
    unary_operator_backward_helper(UnaryOperators::Sigmoid(), a, grad_a, grad_result);
}
void softplus_impl(Tensor a, double beta, Tensor result)
{
    unary_operator_forward_helper(UnaryOperators::Softplus(beta), a, result);
}
void softplus_backward_impl(Tensor a, double beta, Tensor grad_a, Tensor grad_result)
{
    unary_operator_backward_helper(UnaryOperators::Softplus(beta), a, grad_a, grad_result);
}
}  // namespace cuda_impl
}  // namespace tinytorch