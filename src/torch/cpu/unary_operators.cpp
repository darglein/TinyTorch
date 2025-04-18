/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "unary_operators.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops/ops_impl.h"
#include "torch/core/tensor_info.h"
#include "torch/cpu/ops_impl_cpu.h"


namespace tinytorch
{
namespace cpu_impl
{
template <typename T, typename Op>
static void unary_operator_kernel(Op op, TensorInfo<T> a, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;
    auto N  = result.numel();
    if (a.contiguous && result.contiguous)
    {
        T* pa = a.data;
        T* pr = result.data;
#pragma omp parallel for num_threads(get_num_threads())
        for (int64_t i = 0; i < N; ++i)
        {
            G input  = G(pa[i]);
            G output = op.forward(input);
            pr[i]    = T(output);
        }
    }
    else
    {
#pragma omp parallel for num_threads(get_num_threads())
        for (int64_t i = 0; i < N; ++i)
        {
            G input   = G(a[i]);
            G output  = op.forward(input);
            result[i] = T(output);
        }
    }
}
template <typename T, typename Op>
static void unary_operator_backward_kernel(Op op, TensorInfo<T> a, TensorInfo<T> grad_a, TensorInfo<T> grad_result)
{
    using G = typename CpuComputeFloatType<T>::Type;
    auto N  = grad_a.numel();

    if (a.contiguous && grad_a.contiguous && grad_result.contiguous)
    {
        T* pa  = a.data;
        T* pga = grad_a.data;
        T* pgr = grad_result.data;
#pragma omp parallel for num_threads(get_num_threads())
        for (int64_t i = 0; i < N; ++i)
        {
            G input       = G(pa[i]);
            G grad_output = G(pgr[i]);
            G grad_input  = op.backward(input, grad_output);
            pga[i]        = T(grad_input);
        }
    }
    else
    {
#pragma omp parallel for num_threads(get_num_threads())
        for (int64_t i = 0; i < N; ++i)
        {
            G input       = G(a[i]);
            G grad_output = G(grad_result[i]);
            G grad_input  = op.backward(input, grad_output);
            grad_a[i]     = T(grad_input);
        }
    }
}

#define SWITCH_MACRO_UNARY_OPERATOR(op, input, output)                          \
    switch (input.scalar_type())                                                \
    {                                                                           \
        CASE_MACRO((unary_operator_kernel<uint8_t>), kUInt8, op, input, output) \
        CASE_MACRO((unary_operator_kernel<int16_t>), kInt16, op, input, output) \
        CASE_MACRO((unary_operator_kernel<int32_t>), kInt32, op, input, output) \
        CASE_MACRO((unary_operator_kernel<int64_t>), kLong, op, input, output)  \
        CASE_MACRO((unary_operator_kernel<Half>), kHalf, op, input, output)     \
        CASE_MACRO((unary_operator_kernel<float>), kFloat, op, input, output)   \
        CASE_MACRO((unary_operator_kernel<double>), kDouble, op, input, output) \
        default:                                                                \
            CHECK(false) << "invalid input type " << input.scalar_type();       \
    }

#define SWITCH_MACRO_UNARY_OPERATOR_BACKWARD(op, input, grad_input, grad_result)                          \
    switch (input.scalar_type())                                                                          \
    {                                                                                                     \
        CASE_MACRO((unary_operator_backward_kernel<uint8_t>), kUInt8, op, input, grad_input, grad_result) \
        CASE_MACRO((unary_operator_backward_kernel<int16_t>), kInt16, op, input, grad_input, grad_result) \
        CASE_MACRO((unary_operator_backward_kernel<int32_t>), kInt32, op, input, grad_input, grad_result) \
        CASE_MACRO((unary_operator_backward_kernel<int64_t>), kLong, op, input, grad_input, grad_result)  \
        CASE_MACRO((unary_operator_backward_kernel<Half>), kHalf, op, input, grad_input, grad_result)     \
        CASE_MACRO((unary_operator_backward_kernel<float>), kFloat, op, input, grad_input, grad_result)   \
        CASE_MACRO((unary_operator_backward_kernel<double>), kDouble, op, input, grad_input, grad_result) \
        default:                                                                                          \
            CHECK(false) << "invalid input type " << input.scalar_type();                                 \
    }

void abs_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Abs(), a, result);
}
void round_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Round(), a, result);
}
void sqrt_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sqrt(), a, result);
}
void sqrt_backward_impl(Tensor a, Tensor grad_a, Tensor grad_result)
{
    SWITCH_MACRO_UNARY_OPERATOR_BACKWARD(UnaryOperators::Sqrt(), a, grad_a, grad_result);
}

void log_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Log(), a, result);
}
void exp_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Exp(), a, result);
}
void sign_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sign(), a, result);
}
void sin_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sin(), a, result);
}
void cos_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Cos(), a, result);
}
void relu_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Relu(), a, result);
}
void sigmoid_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sigmoid(), a, result);
}
void sigmoid_backward_impl(Tensor a, Tensor grad_a, Tensor grad_result)
{
    SWITCH_MACRO_UNARY_OPERATOR_BACKWARD(UnaryOperators::Sigmoid(), a, grad_a, grad_result);
}

void softplus_impl(Tensor a, double beta, Tensor result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Softplus((float)beta), a, result);
}
void softplus_backward_impl(Tensor a, double beta, Tensor grad_a, Tensor grad_result)
{
    SWITCH_MACRO_UNARY_OPERATOR_BACKWARD(UnaryOperators::Softplus((float)beta), a, grad_a, grad_result);
}

}  // namespace cpu_impl

}  // namespace tinytorch
