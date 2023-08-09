/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "binary_operators.h"

#include "torch/core/ops.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/tensor_info.h"
#include "torch/cpu/ops_impl_cpu.h"


#define SWITCH_MACRO_ALL_OPERATOR(real_scalar_type, op, func, ...)      \
    switch (real_scalar_type)                                           \
    {                                                                   \
        CASE_MACRO((func<uint8_t>), kUInt8, op<uint8_t>(), __VA_ARGS__) \
        CASE_MACRO((func<int16_t>), kInt16, op<int16_t>(), __VA_ARGS__) \
        CASE_MACRO((func<int32_t>), kInt32, op<int32_t>(), __VA_ARGS__) \
        CASE_MACRO((func<int64_t>), kLong, op<int64_t>(), __VA_ARGS__)  \
        CASE_MACRO((func<Half>), kHalf, op<float>(), __VA_ARGS__)       \
        CASE_MACRO((func<float>), kFloat, op<float>(), __VA_ARGS__)     \
        CASE_MACRO((func<double>), kDouble, op<double>(), __VA_ARGS__)  \
        default:                                                        \
            CHECK(false) << "invalid input type " << real_scalar_type;  \
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
        result[index_result] = op(a[index_a], b[index_b]);
    }
}
template <typename T, typename Op>
static void element_wise_operator(Op op, TensorInfo<T> a, T b, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        result[i] = T(G(op(G(a[i]), b)));
    }
}
template <typename T, typename Op>
static void element_wise_operator(Op op, T a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        result[i] = op(a, b[i]);
    }
}

void add_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::plus, element_wise_operator, a, b, result);
}
void add_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::plus, element_wise_operator, a, b, result);
}

void sub_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::minus, element_wise_operator, a, b, result);
}
void sub_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::minus, element_wise_operator, a, b, result);
}

void mult_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::multiplies, element_wise_operator, a, b, result);
}
void mult_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::multiplies, element_wise_operator, a, b, result);
}
void div_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::divides, element_wise_operator, a, b, result);
}
void div_impl(double a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(b.scalar_type(), std::divides, element_wise_operator, a, b, result);
}
void equal_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::equal_to, element_wise_operator, a, b, result);
}
void less_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::less, element_wise_operator, a, b, result);
}
void greater_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), std::greater, element_wise_operator, a, b, result);
}

template <typename T>
static void pow_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(std::pow(G(a[i]), G(b)));
    }
}

void pow_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), pow_impl, a, b, result);
}

template <typename T>
static void pow_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(std::pow(G(a[i]), G(b[i])));
    }
}

void pow_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), pow_impl, a, b, result);
}



template <typename T>
static void print_impl(std::ostream& strm, TensorInfo<T> a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        strm << a[i] << " ";
    }
}

void print_impl(std::ostream& strm, Tensor t)
{
    print_impl<float>(strm, t);
}



}  // namespace cpu_impl

}  // namespace tinytorch