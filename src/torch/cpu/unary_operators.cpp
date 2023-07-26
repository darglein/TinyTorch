/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "unary_operators.h"

#include "torch/core/ops.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/tensor_info.h"
#include "torch/cpu/ops_impl_cpu.h"
#include "torch/core/ops_impl_shared.h"


namespace tinytorch
{
namespace cpu_impl
{

template <typename T>
static void sqrt_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(std::sqrt(a[i]));
    }
}

void sqrt_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sqrt_impl, a, result);
}

template <typename T>
static void log_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::log(a[i]);
    }
}

void log_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), log_impl, a, result);
}

template <typename T>
static void log1p_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::log1p(a[i]);
    }
}

void log1p_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), log1p_impl, a, result);
}

template <typename T>
static void exp_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::exp(a[i]);
    }
}

void exp_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), exp_impl, a, result);
}

template <typename T>
static void sign_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        T v       = a[i];
        result[i] = (v < T(0)) ? T(-1) : (v > T(0)) ? T(1) : T(0);
    }
}

void sign_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sign_impl, a, result);
}

template <typename T>
static void pow_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(std::pow(a[i], b));
    }
}

void pow_impl(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), pow_impl, a, b, result);
}

template <typename T>
static void sin_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::sin(a[i]);
    }
}

void sin_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sin_impl, a, result);
}

template <typename T>
static void cos_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::cos(a[i]);
    }
}

void cos_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), cos_impl, a, result);
}

template <typename T>
static void relu_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = relu(a[i]);
    }
}

void relu_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), relu_impl, a, result);
}

template <typename T>
static void sigmoid_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = sigmoid(a[i]);
    }
}

void sigmoid_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sigmoid_impl, a, result);
}

template <typename T>
static void softplus_impl(TensorInfo<T> a, double beta, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = softplus(a[i], T(beta));
    }
}

void softplus_impl(Tensor a, double beta, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), softplus_impl, a, beta, result);
}




template <typename T>
static void log_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = log_backward(a[i]) * grad_output[i];
    }
}

void log_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), log_backward_impl, a, grad_output, grad_a);
}

template <typename T>
static void log1p_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = log1p_backward(a[i]) * grad_output[i];
    }
}

void log1p_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), log1p_backward_impl, a, grad_output, grad_a);
}

template <typename T>
static void exp_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = std::exp(a[i]) * grad_output[i];
    }
}

void exp_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), exp_backward_impl, a, grad_output, grad_a);
}

void sign_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    throw std::runtime_error("not implemented");
}

template <typename T>
static void pow_backward_impl(TensorInfo<T> a, double b, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = pow_backward(a[i], T(b)) * grad_output[i];
    }
}

void pow_backward_impl(Tensor a, double b, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), pow_backward_impl, a, b, grad_output, grad_a);
}

template <typename T>
static void sin_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = sin_backward(a[i]) * grad_output[i];
    }
}

void sin_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), sin_backward_impl, a, grad_output, grad_a);
}

template <typename T>
static void cos_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = cos_backward(a[i]) * grad_output[i];
    }
}

void cos_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), cos_backward_impl, a, grad_output, grad_a);
}

template <typename T>
static void relu_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = relu_backward(a[i]) * grad_output[i];
    }
}

void relu_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), relu_backward_impl, a, grad_output, grad_a);
}

template <typename T>
static void sigmoid_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = sigmoid_backward(a[i]) * grad_output[i];
    }
}

void sigmoid_backward_impl(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), sigmoid_backward_impl, a, grad_output, grad_a);
}

template <typename T>
static void softplus_backward_impl(TensorInfo<T> a, double beta, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = softplus_backward(a[i], T(beta)) * grad_output[i];
    }
}

void softplus_backward_impl(Tensor a, double beta, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), softplus_backward_impl, a, beta, grad_output, grad_a);
}

}  // namespace cpu_impl

}  // namespace tinytorch
