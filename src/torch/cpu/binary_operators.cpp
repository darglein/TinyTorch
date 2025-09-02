/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops/ops_impl.h"
#include "torch/core/tensor_info.h"
#include "torch/cpu/ops_impl_cpu.h"


#define SWITCH_MACRO_BINARY_OPERATOR(op, a, b, result)                           \
    switch (result.scalar_type())                                                \
    {                                                                            \
        CASE_MACRO((element_wise_operator<uint8_t, false>), kUInt8, op, a, b, result)   \
        CASE_MACRO((element_wise_operator<uint16_t, false>), kUInt16, op, a, b, result) \
        CASE_MACRO((element_wise_operator<int16_t, false>), kInt16, op, a, b, result)   \
        CASE_MACRO((element_wise_operator<int32_t, false>), kInt32, op, a, b, result)   \
        CASE_MACRO((element_wise_operator<int64_t, false>), kLong, op, a, b, result)    \
        CASE_MACRO((element_wise_operator<Half, false>), kHalf, op, a, b, result)       \
        CASE_MACRO((element_wise_operator<float, false>), kFloat, op, a, b, result)     \
        CASE_MACRO((element_wise_operator<double, false>), kDouble, op, a, b, result)   \
        default:                                                                 \
            CHECK(false) << "invalid input type " << result.scalar_type();       \
    }

#define SWITCH_MACRO_BINARY_OPERATOR_VEC(op, a, b, result)                           \
    switch (result.scalar_type())                                                \
    {                                                                            \
        CASE_MACRO((element_wise_operator<uint8_t, true>), kUInt8, op, a, b, result)   \
        CASE_MACRO((element_wise_operator<uint16_t, true>), kUInt16, op, a, b, result) \
        CASE_MACRO((element_wise_operator<int16_t, true>), kInt16, op, a, b, result)   \
        CASE_MACRO((element_wise_operator<int32_t, true>), kInt32, op, a, b, result)   \
        CASE_MACRO((element_wise_operator<int64_t, true>), kLong, op, a, b, result)    \
        CASE_MACRO((element_wise_operator<Half, true>), kHalf, op, a, b, result)       \
        CASE_MACRO((element_wise_operator<float, true>), kFloat, op, a, b, result)     \
        CASE_MACRO((element_wise_operator<double, true>), kDouble, op, a, b, result)   \
        default:                                                                 \
            CHECK(false) << "invalid input type " << result.scalar_type();       \
    }


namespace tinytorch
{
namespace cpu_impl
{

template <typename T, bool vectorize, typename Op>
static void element_wise_operator(Op op, TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    if (a.numel() == b.numel() && a.contiguous && b.contiguous)
    {
        // std::cout << "test" << std::endl;
        // fast implementation for contiguous case (without dim expansion)
        const T* __restrict__ pa  = a.data;
        const T* __restrict__ pb  = b.data;
        T* __restrict__ pr  = result.data;
        auto N = result.numel();

#pragma omp simd if(vectorize)
        for (int64_t i = 0; i < N; ++i)
        {
            pr[i] = op.forward(pa[i], pb[i]);
        }
    }
    else
    {
#pragma omp parallel for num_threads(get_num_threads())
        for (int64_t i = 0; i < result.numel(); ++i)
        {
            auto index_result = result.LinearIndexToDimIndex(i);
            // the index clamping allows operations when one tensor has a 1-dimension
            auto index_a         = a.clamp_index_to_size(index_result);
            auto index_b         = b.clamp_index_to_size(index_result);
            result[index_result] = op.forward(a[index_a], b[index_b]);
        }
    }
}
template <typename T, bool vectorize, typename Op>
static void element_wise_operator(Op op, TensorInfo<T> a, T b, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;

    if (a.contiguous && result.contiguous)
    {
        // fast implementation for contiguous case (without dim expansion)
        T* pa  = a.data;
        T* pr  = result.data;
        auto N = result.numel();
#pragma omp simd if(vectorize)
        for (int64_t i = 0; i < N; ++i)
        {
            pr[i] = T(G(op.forward(G(pa[i]), G(b))));
        }
    }
    else
    {
#pragma omp parallel for num_threads(get_num_threads())
        for (int64_t i = 0; i < result.numel(); ++i)
        {
            result[i] = T(G(op.forward(G(a[i]), G(b))));
        }
    }
}
template <typename T, bool vectorize, typename Op>
static void element_wise_operator(Op op, T a, TensorInfo<T> b, TensorInfo<T> result)
{
    if (b.contiguous && result.contiguous)
    {
        // fast implementation for contiguous case (without dim expansion)
        T* pb  = b.data;
        T* pr  = result.data;
        auto N = result.numel();
#pragma omp simd if(vectorize)
        for (int64_t i = 0; i < N; ++i)
        {
            pr[i] = op.forward(a, pb[i]);
        }
    }
    else
    {
#pragma omp parallel for num_threads(get_num_threads())
        for (int64_t i = 0; i < result.numel(); ++i)
        {
            result[i] = op.forward(a, b[i]);
        }
    }
}

#ifdef _MSC_VER
#    pragma warning( \
        disable : 4244)  // warning C4244: 'argument': conversion from 'double' to 'T', possible loss of data
#endif

void add_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Add(), a, b, result);
}
void add_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Add(), a, b, result);
}

void sub_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Sub(), a, b, result);
}
void sub_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Sub(), a, b, result);
}
void mult_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Mult(), a, b, result);
}
void mult_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Mult(), a, b, result);
}
void div_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Div(), a, b, result);
}
void div_impl(double a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Div(), a, b, result);
}
void equal_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Equal(), a, b, result);
}
void less_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Less(), a, b, result);
}
void greater_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Greater(), a, b, result);
}
void min_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Min(), a, b, result);
}
void max_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR_VEC(BinaryOperators::Max(), a, b, result);
}
void pow_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Pow(), a, b, result);
}
void pow_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Pow(), a, b, result);
}

}  // namespace cpu_impl

}  // namespace tinytorch