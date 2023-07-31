/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops.h"

#include "unary_operators.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{

namespace cuda_impl
{


template <typename T>
__launch_bounds__(128) static __global__ void abs_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = ::abs(a[i]);
}

void abs_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), abs_impl, a, result);
}


template <typename T>
__launch_bounds__(128) static __global__ void sqrt_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = ::sqrt(a[i]);
}

void sqrt_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sqrt_impl, a, result);
}


template <typename T>
__launch_bounds__(128) static __global__ void log_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = ::log(a[i]);
}

void log_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), log_impl, a, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void exp_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = ::exp(a[i]);
}

void exp_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), exp_impl, a, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void sign_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    T v       = a[i];
    result[i] = (v < T(0)) ? T(-1) : (v > T(0)) ? T(1) : T(0);
}

void sign_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sign_impl, a, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void pow_impl(TensorInfoCuda<T> a, double b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = T(::pow(a[i], b));
}

void pow_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), pow_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void sin_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = ::sin(a[i]);
}

void sin_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sin_impl, a, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void cos_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = ::cos(a[i]);
}

void cos_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), cos_impl, a, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void relu_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = relu(a[i]);
}

void relu_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), relu_impl, a, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void sigmoid_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = sigmoid(a[i]);
}

void sigmoid_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sigmoid_impl, a, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void softplus_impl(TensorInfoCuda<T> a, double beta, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = softplus(a[i], T(beta));
}

void softplus_impl(Tensor a, double beta, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), softplus_impl, a, beta, result);
}

}  // namespace cuda_impl
}  // namespace tinytorch