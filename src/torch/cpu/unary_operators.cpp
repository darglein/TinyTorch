/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "unary_operators.h"

#include "torch/core/ops.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops_impl_shared.h"
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
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        G input   = G(a[i]);
        G output  = op.forward(input);
        result[i] = T(output);
    }
}

#define SWITCH_MACRO_UNARY_OPERATOR(op, input, output)                          \
    switch (input.scalar_type())                                                \
    {                                                                           \
        CASE_MACRO((unary_operator_kernel<uint8_t>), kUInt8, op, input, output) \
        CASE_MACRO((unary_operator_kernel<int16_t>), kInt16, op, input, output) \
        CASE_MACRO((unary_operator_kernel<int32_t>), kInt32, op, input, output) \
        CASE_MACRO((unary_operator_kernel<int64_t>), kLong, op, input, output)  \
        CASE_MACRO((unary_operator_kernel<Half>), kHalf, op, input, output)   \
        CASE_MACRO((unary_operator_kernel<float>), kFloat, op, input, output)   \
        CASE_MACRO((unary_operator_kernel<double>), kDouble, op, input, output) \
        default:                                                                \
            CHECK(false) << "invalid input type " << input.scalar_type();       \
    }


void abs_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Abs(), a, result);
}
void round_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Round(), a, result);
}
void sqrt_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sqrt(), a, result);
}
void log_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Log(), a, result);
}
void exp_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Exp(), a, result);
}
void sign_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sign(), a, result);
}
void sin_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sin(), a, result);
}
void cos_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Cos(), a, result);
}
void relu_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Relu(), a, result);
}
void sigmoid_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Sigmoid(), a, result);
}
void softplus_impl(Tensor a, double beta, Tensor& result)
{
    SWITCH_MACRO_UNARY_OPERATOR(UnaryOperators::Softplus(beta), a, result);
}


}  // namespace cpu_impl

}  // namespace tinytorch
