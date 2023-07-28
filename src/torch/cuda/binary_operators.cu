#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/binary_operators.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>


#define CUDA_SWITCH_MACRO_ALL_OPERATOR(real_scalar_type, numel, op, func, ...)      \
    switch (real_scalar_type)                                                       \
    {                                                                               \
        CUDA_CASE_MACRO((func<uint8_t>), kUInt8, numel, op<uint8_t>(), __VA_ARGS__) \
        CUDA_CASE_MACRO((func<int16_t>), kInt16, numel, op<int16_t>(), __VA_ARGS__) \
        CUDA_CASE_MACRO((func<int32_t>), kInt32, numel, op<int32_t>(), __VA_ARGS__) \
        CUDA_CASE_MACRO((func<int64_t>), kLong, numel, op<int64_t>(), __VA_ARGS__)  \
        CUDA_CASE_MACRO((func<float>), kFloat, numel, op<float>(), __VA_ARGS__)     \
        CUDA_CASE_MACRO((func<double>), kDouble, numel, op<double>(), __VA_ARGS__)  \
        default:                                                                    \
            CHECK(false) << "invalid input type " << real_scalar_type;              \
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
    result[index_result] = op(a[index_a], b[index_b]);
}

template <typename T, typename Op>
__launch_bounds__(128) __global__ static void element_wise_operator(Op op, TensorInfoCuda<T> a, T b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    result[i] = op(a[i], b);
}
template <typename T, typename Op>
__launch_bounds__(128) __global__ static void element_wise_operator(Op op, T a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    result[i] = op(a, b[i]);
}


void add_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::plus, element_wise_operator, a, b, result);
}
void add_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::plus, element_wise_operator, a, b, result);
}

void sub_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::minus, element_wise_operator, a, b, result);
}
void sub_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::minus, element_wise_operator, a, b, result);
}

void mult_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::multiplies, element_wise_operator, a, b, result);
}
void mult_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::multiplies, element_wise_operator, a, b, result);
}
void div_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::divides, element_wise_operator, a, b, result);
}
void div_impl(double a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(b.scalar_type(), result.numel(), std::divides, element_wise_operator, a, b, result);
}
void equal_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::equal_to, element_wise_operator, a, b, result);
}
void less_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::less, element_wise_operator, a, b, result);
}
void greater_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL_OPERATOR(a.scalar_type(), result.numel(), std::greater, element_wise_operator, a, b, result);
}


}  // namespace cuda_impl
}  // namespace tinytorch