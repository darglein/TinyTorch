/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops.h"

#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
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
    if (i >= result.numel()) return;
    {
        G input   = G(a[i]);
        G output  = op.forward(input);
        result[i] = T(output);
    }
}

#define SWITCH_MACRO_UNARY_OPERATOR(op, input, output)                                              \
    switch (input.scalar_type())                                                                    \
    {                                                                                               \
        CUDA_CASE_MACRO((unary_operator_kernel<uint8_t>), kUInt8, input.numel(), op, input, output) \
        CUDA_CASE_MACRO((unary_operator_kernel<int16_t>), kInt16, input.numel(), op, input, output) \
        CUDA_CASE_MACRO((unary_operator_kernel<int32_t>), kInt32, input.numel(), op, input, output) \
        CUDA_CASE_MACRO((unary_operator_kernel<int64_t>), kLong, input.numel(), op, input, output)  \
        CUDA_CASE_MACRO((unary_operator_kernel<__half>), kHalf, input.numel(), op, input, output)   \
        CUDA_CASE_MACRO((unary_operator_kernel<float>), kFloat, input.numel(), op, input, output)   \
        CUDA_CASE_MACRO((unary_operator_kernel<double>), kDouble, input.numel(), op, input, output) \
        default:                                                                                    \
            CHECK(false) << "invalid input type " << input.scalar_type();                           \
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

}  // namespace cuda_impl
}  // namespace tinytorch