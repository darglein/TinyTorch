/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops/ops_impl.h"
#include "torch/core/tensor_info.h"
#include "torch/cpu/ops_impl_cpu.h"


#define SWITCH_MACRO_BINARY_OPERATOR(op, a, b, result)                         \
    switch (result.scalar_type())                                              \
    {                                                                          \
        CASE_MACRO((element_wise_operator<uint8_t>), kUInt8, op, a, b, result) \
        CASE_MACRO((element_wise_operator<int16_t>), kInt16, op, a, b, result) \
        CASE_MACRO((element_wise_operator<int32_t>), kInt32, op, a, b, result) \
        CASE_MACRO((element_wise_operator<int64_t>), kLong, op, a, b, result)  \
        CASE_MACRO((element_wise_operator<Half>), kHalf, op, a, b, result)     \
        CASE_MACRO((element_wise_operator<float>), kFloat, op, a, b, result)   \
        CASE_MACRO((element_wise_operator<double>), kDouble, op, a, b, result) \
        default:                                                               \
            CHECK(false) << "invalid input type " << result.scalar_type();     \
    }


namespace tinytorch
{
namespace cpu_impl
{

template <typename T, typename Op>
static void element_wise_operator(Op op, TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto index_result = result.LinearIndexToDimIndex(i);
        // the index clamping allows operations when one tensor has a 1-dimension
        auto index_a         = a.clamp_index_to_size(index_result);
        auto index_b         = b.clamp_index_to_size(index_result);
        result[index_result] = op.forward(a[index_a], b[index_b]);
    }
}
template <typename T, typename Op>
static void element_wise_operator(Op op, TensorInfo<T> a, T b, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        result[i] = T(G(op.forward(G(a[i]), G(b))));
    }
}
template <typename T, typename Op>
static void element_wise_operator(Op op, T a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        result[i] = op.forward(a, b[i]);
    }
}


void add_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Add(), a, b, result);
}
void add_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Add(), a, b, result);
}

void sub_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Sub(), a, b, result);
}
void sub_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Sub(), a, b, result);
}
void mult_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Mult(), a, b, result);
}
void mult_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Mult(), a, b, result);
}
void div_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Div(), a, b, result);
}
void div_impl(double a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Div(), a, b, result);
}
void equal_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Equal(), a, b, result);
}
void less_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Less(), a, b, result);
}
void greater_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Greater(), a, b, result);
}
void pow_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Pow(), a, b, result);
}
void pow_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Pow(), a, b, result);
}
void min_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Min(), a, b, result);
}
void max_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Max(), a, b, result);
}


}  // namespace cpu_impl

}  // namespace tinytorch