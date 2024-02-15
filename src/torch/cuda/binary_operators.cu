#include "torch/core/ops/ops_impl.h"

#include "torch/cuda/binary_operators.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>


#define SWITCH_MACRO_BINARY_OPERATOR(op, a, b, result)                                              \
    switch (result.scalar_type())                                                                   \
    {                                                                                               \
        CUDA_CASE_MACRO(element_wise_operator<uint8_t>, kUInt8, result.numel(), op, a, b, result) \
        CUDA_CASE_MACRO(element_wise_operator<int16_t>, kInt16, result.numel(), op, a, b, result) \
        CUDA_CASE_MACRO(element_wise_operator<int32_t>, kInt32, result.numel(), op, a, b, result) \
        CUDA_CASE_MACRO(element_wise_operator<int64_t>, kInt64, result.numel(), op, a, b, result) \
        CUDA_CASE_MACRO(element_wise_operator<half>, kHalf, result.numel(), op, a, b, result)     \
        CUDA_CASE_MACRO(element_wise_operator<float>, kFloat, result.numel(), op, a, b, result)   \
        CUDA_CASE_MACRO(element_wise_operator<double>, kDouble, result.numel(), op, a, b, result) \
        default:                                                                                    \
            CHECK(false) << "invalid input type " << result.scalar_type();                          \
    }


namespace tinytorch
{
namespace cuda_impl
{

template <typename T, typename Op>
__launch_bounds__(128) __global__
    static void element_wise_operator(Op op, TensorInfoCuda<T> a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    auto index_result = result.LinearIndexToDimIndex(i);
    // the index clamping allows operations when one tensor has a 1-dimension
     auto index_a         = a.clamp_index_to_size(index_result);
     auto index_b         = b.clamp_index_to_size(index_result);
    result[index_result] = op.forward(a[index_a], b[index_b]);
}

template <typename T, typename Op>
__launch_bounds__(128) __global__
    static void element_wise_operator(Op op, TensorInfoCuda<T> a, T b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    result[i] = op.forward(a[i], b);
}
template <typename T, typename Op>
__launch_bounds__(128) __global__
    static void element_wise_operator(Op op, T a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    result[i] = op.forward(a, b[i]);
}

#ifdef _MSC_VER
#pragma warning( disable : 4244 ) // warning C4244: 'argument': conversion from 'double' to 'T', possible loss of data
#endif

void add_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Add(), a, b, result);
}
void add_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Add(), a, b, result);
}
void sub_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Sub(), a, b, result);
}
void sub_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Sub(), a, b, result);
}
void mult_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Mult(), a, b, result);
}
void mult_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Mult(), a, b, result);
}
void div_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Div(), a, b, result);
}
void div_impl(double a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Div(), a, b, result);
}
void equal_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Equal(), a, b, result);
}
void less_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Less(), a, b, result);
}
void greater_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Greater(), a, b, result);
}
void pow_impl(Tensor a, double b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Pow(), a, b, result);
}
void pow_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Pow(), a, b, result);
}
void min_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Min(), a, b, result);
}
void max_impl(Tensor a, Tensor b, Tensor result)
{
    SWITCH_MACRO_BINARY_OPERATOR(BinaryOperators::Max(), a, b, result);
}

}  // namespace cuda_impl
}  // namespace tinytorch