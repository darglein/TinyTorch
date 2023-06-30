/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops.h"
#include "torch/core/tensor.h"
#include "torch/cpu/ops_impl_cpu.h"
#include "torch/core/tensor_info.h"
#include "torch/core/ops_impl_shared.h"

namespace TINY_TORCH_NAMESPACE
{

#define CASE_MACRO(func, type, scalar_type, ...) \
    case scalar_type:                            \
        func<type>(__VA_ARGS__);                 \
        break;

#define SWITCH_MACRO_FLOAT(real_scalar_type, func, ...) \
    switch (real_scalar_type)                           \
    {                                                   \
        CASE_MACRO(func, float, kFloat, __VA_ARGS__)    \
        CASE_MACRO(func, double, kDouble, __VA_ARGS__)  \
        default:                                        \
            assert(false);                              \
    }

// TODO: Half!
#define SWITCH_MACRO_ALL(real_scalar_type, func, ...)  \
    switch (real_scalar_type)                          \
    {                                                  \
        CASE_MACRO(func, uint8_t, kUInt8, __VA_ARGS__) \
        CASE_MACRO(func, int16_t, kInt16, __VA_ARGS__) \
        CASE_MACRO(func, int32_t, kInt32, __VA_ARGS__) \
        CASE_MACRO(func, int64_t, kLong, __VA_ARGS__)  \
        CASE_MACRO(func, float, kFloat, __VA_ARGS__)   \
        CASE_MACRO(func, double, kDouble, __VA_ARGS__) \
        default:                                       \
            assert(false);                             \
    }

template <typename T>
static void fill_impl_cpu(TensorInfo<T> a, double value)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = T(value);
    }
}

void fill_impl_cpu(Tensor a, double value)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), fill_impl_cpu, a, value);
}

template <typename T>
static void square_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto v    = a[i];
        result[i] = v * v;
    }
}

Tensor square_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), square_impl_cpu, a, result);
    return result;
}

template <typename T>
static void add_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] + b[i];
    }
}

Tensor add_impl_cpu(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void add_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] + b);
    }
}

Tensor add_impl_cpu(Tensor a, double b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void add_impl_cpu(double a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < b.numel(); ++i)
    {
        result[i] = T(a + b[i]);
    }
}

Tensor add_impl_cpu(double a, Tensor b)
{
    Tensor result = empty_like(b);
    SWITCH_MACRO_ALL(b.scalar_type(), add_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void sub_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] - b[i];
    }
}

Tensor sub_impl_cpu(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void sub_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] - b);
    }
}

Tensor sub_impl_cpu(Tensor a, double b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void sub_impl_cpu(double a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < b.numel(); ++i)
    {
        result[i] = T(a - b[i]);
    }
}

Tensor sub_impl_cpu(double a, Tensor b)
{
    Tensor result = empty_like(b);
    SWITCH_MACRO_ALL(b.scalar_type(), sub_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void mult_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] * b[i];
    }
}

Tensor mult_impl_cpu(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void mult_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] * b);
    }
}

Tensor mult_impl_cpu(Tensor a, double b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void mult_impl_cpu(double a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < b.numel(); ++i)
    {
        result[i] = T(a * b[i]);
    }
}

Tensor mult_impl_cpu(double a, Tensor b)
{
    Tensor result = empty_like(b);
    SWITCH_MACRO_ALL(b.scalar_type(), mult_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void div_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] / b[i];
    }
}

Tensor div_impl_cpu(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void div_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] / b);
    }
}

Tensor div_impl_cpu(Tensor a, double b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void neg_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = -a[i];
    }
}

Tensor neg_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), neg_impl_cpu, a, result);
    return result;
}

template <typename T>
static void sum_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] += a[i];
    }
}

Tensor sum_impl_cpu(Tensor a)
{
    Tensor result = zeros({1});
    SWITCH_MACRO_ALL(a.scalar_type(), sum_impl_cpu, a, result);
    return result;
}

template <typename T>
static void log_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::log(a[i]);
    }
}

Tensor log_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), log_impl_cpu, a, result);
    return result;
}

template <typename T>
static void log1p_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::log1p(a[i]);
    }
}

Tensor log1p_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), log1p_impl_cpu, a, result);
    return result;
}

template <typename T>
static void exp_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::exp(a[i]);
    }
}

Tensor exp_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), exp_impl_cpu, a, result);
    return result;
}

template <typename T>
static void sign_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        T v = a[i];
        result[i] = (v < T(0)) ? T(-1) : (v > T(0)) ? T(1) : T(0);
    }
}

Tensor sign_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), sign_impl_cpu, a, result);
    return result;
}

template <typename T>
static void pow_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(std::pow(a[i], b));
    }
}

Tensor pow_impl_cpu(Tensor a, double b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), pow_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void sin_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::sin(a[i]);
    }
}

Tensor sin_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), sin_impl_cpu, a, result);
    return result;
}

template <typename T>
static void cos_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::cos(a[i]);
    }
}

Tensor cos_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), cos_impl_cpu, a, result);
    return result;
}

template <typename T>
static void relu_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = relu(a[i]);
    }
}

Tensor relu_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), relu_impl_cpu, a, result);
    return result;
}

template <typename T>
static void sigmoid_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = sigmoid(a[i]);
    }
}

Tensor sigmoid_impl_cpu(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), sigmoid_impl_cpu, a, result);
    return result;
}

template <typename T>
static void softplus_impl_cpu(TensorInfo<T> a, double beta, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = softplus(a[i], T(beta));
    }
}

Tensor softplus_impl_cpu(Tensor a, double beta)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), softplus_impl_cpu, a, beta, result);
    return result;
}

template <typename T>
static void prod_impl_cpu(TensorInfo<T> a, int64_t dim, TensorInfo<T> result)
{
    throw std::runtime_error("not implemented");
}

Tensor prod_impl_cpu(Tensor a, int64_t dim)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), prod_impl_cpu, a, dim, result);
    return result;
}

template <typename T>
static void min_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::min(a[i], b[i]);
    }
}

Tensor min_impl_cpu(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void max_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::max(a[i], b[i]);
    }
}

Tensor max_impl_cpu(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl_cpu, a, b, result);
    return result;
}

template <typename T>
static void index_select_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<int64_t> index, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    int64_t to_copy = input.numel() / input.sizes[dim];
    for (int64_t index_index = 0; index_index < index.numel(); ++index_index)
    {
        int64_t slice = index[index_index];
        int64_t input_start = slice * input.strides[dim];
        int64_t result_start = index_index * result.strides[dim];
        
        for (int64_t c = 0; c < to_copy; ++c)
        {
            int64_t linearId = c;

            int64_t input_offset = input_start;
            int64_t result_offset = result_start;
            for (int64_t i = dims - 1; i > 0; --i)
            {
                if (i != dim)
                {
                    int64_t curDimIndex  = linearId % input.sizes[i];
                    input_offset += curDimIndex * input.strides[i];
                    result_offset += curDimIndex * result.strides[i];
                    linearId /= input.sizes[i];
                }
            }

            if (dim != 0)
            {
                input_offset += linearId * input.strides[0];
                result_offset += linearId * result.strides[0];
            }

            result.data[result_offset] = input.data[input_offset];
        }
    }
}

Tensor index_select_impl_cpu(Tensor input, int64_t dim, Tensor index)
{
    assert(dim < input.dim());
    assert(index.dtype() == kLong);

    auto numel = index.numel();

    auto result_size = input.sizes().vec();
    result_size[dim] = numel;

    Tensor result = empty(result_size, input.options());

    SWITCH_MACRO_ALL(input.scalar_type(), index_select_impl_cpu, input, dim, index, result);

    return result;
}

// ================================================================================================================

template <typename T>
static void square_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        grad_a[i] = 2 * a[i] * grad_output[i];
    }
}

std::vector<Tensor> square_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), square_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}


// Function for when the derivative is one.
template <typename T>
static void one_backward_impl_cpu(TensorInfo<T> grad_output, TensorInfo<T> grad_a, TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < grad_output.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = g;
        grad_b[i] = g;
    }
}

std::vector<Tensor> add_backward_impl_cpu(Tensor grad_output)
{
    Tensor grad_a = empty_like(grad_output);
    Tensor grad_b = empty_like(grad_output);
    SWITCH_MACRO_ALL(grad_output.scalar_type(), one_backward_impl_cpu, grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}

template <typename T>
static void sub_backward_impl_cpu(TensorInfo<T> grad_output, TensorInfo<T> grad_a, TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < grad_output.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = g;
        grad_b[i] = -g;
    }
}

std::vector<Tensor> sub_backward_impl_cpu(Tensor grad_output)
{
    Tensor grad_a = empty_like(grad_output);
    Tensor grad_b = empty_like(grad_output);
    SWITCH_MACRO_ALL(grad_output.scalar_type(), sub_backward_impl_cpu, grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}


template <typename T>
static void mult_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> grad_output, TensorInfo<T> grad_a,
                               TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = b[i] * g;
        grad_b[i] = a[i] * g;
    }
}

std::vector<Tensor> mult_backward_impl_cpu(Tensor a, Tensor b, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    Tensor grad_b = empty_like(b);
    SWITCH_MACRO_ALL(a.scalar_type(), mult_backward_impl_cpu, a, b, grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}

template <typename T>
static void mult_backward_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = T(b * g);
    }
}

std::vector<Tensor> mult_backward_impl_cpu(Tensor a, double b, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), mult_backward_impl_cpu, a, b, grad_output, grad_a);
    return {grad_a};
}

std::vector<Tensor> mult_backward_impl_cpu(double b, Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), mult_backward_impl_cpu, a, b, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void div_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> grad_output, TensorInfo<T> grad_a,
                              TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto g    = grad_output[i];
        auto [ga, gb] = div_backward(a[i], b[i]);
        grad_a[i] = ga * g;
        grad_b[i] = gb * g;
    }
}

std::vector<Tensor> div_backward_impl_cpu(Tensor a, Tensor b, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    Tensor grad_b = empty_like(b);
    SWITCH_MACRO_ALL(a.scalar_type(), div_backward_impl_cpu, a, b, grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}

template <typename T>
static void div_backward_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto g    = grad_output[i];
        auto [ga, gb] = div_backward(a[i], T(b));
        grad_a[i] = ga * g;
    }
}

std::vector<Tensor> div_backward_impl_cpu(Tensor a, double b, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), div_backward_impl_cpu, a, b, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void div_backward_impl_cpu(double a, TensorInfo<T> b, TensorInfo<T> grad_output, TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < b.numel(); ++i)
    {
        auto g    = grad_output[i];
        auto [ga, gb] = div_backward(T(a), b[i]);
        grad_b[i] = gb * g;
    }
}

std::vector<Tensor> div_backward_impl_cpu(double a, Tensor b, Tensor grad_output)
{
    Tensor grad_b = empty_like(b);
    SWITCH_MACRO_ALL(b.scalar_type(), div_backward_impl_cpu, a, b, grad_output, grad_b);
    return {grad_b};
}

template <typename T>
static void neg_backward_impl_cpu(TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = -grad_output[i];
    }
}

std::vector<Tensor> neg_backward_impl_cpu(Tensor grad_output)
{
    Tensor grad_a = empty_like(grad_output);
    SWITCH_MACRO_ALL(grad_output.scalar_type(), neg_backward_impl_cpu, grad_output, grad_a);
    return {grad_a};
}



template <typename T>
static void sum_backward_impl_cpu(TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    auto g = grad_output[0];
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = g;
    }
}

std::vector<Tensor> sum_backward_impl_cpu(const SizeType& input_sizes, Tensor grad_output)
{
    assert(grad_output.numel() == 1);
    Tensor grad_a = empty(input_sizes);
    SWITCH_MACRO_ALL(grad_output.scalar_type(), sum_backward_impl_cpu, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void log_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = log_backward(a[i]) * grad_output[i];
    }
}

std::vector<Tensor> log_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), log_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void log1p_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = log1p_backward(a[i]) * grad_output[i];
    }
}

std::vector<Tensor> log1p_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), log1p_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void exp_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = std::exp(a[i]) * grad_output[i];
    }
}

std::vector<Tensor> exp_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), exp_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}

std::vector<Tensor> sign_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    throw std::runtime_error("not implemented");
    return {};
}

template <typename T>
static void pow_backward_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = pow_backward(a[i], T(b)) * grad_output[i];
    }
}

std::vector<Tensor> pow_backward_impl_cpu(Tensor a, double b, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), pow_backward_impl_cpu, a, b, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void sin_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = sin_backward(a[i]) * grad_output[i];
    }
}

std::vector<Tensor> sin_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), sin_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void cos_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = cos_backward(a[i]) * grad_output[i];
    }
}

std::vector<Tensor> cos_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), cos_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void relu_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = relu_backward(a[i]) * grad_output[i];
    }
}

std::vector<Tensor> relu_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), relu_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void sigmoid_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = sigmoid_backward(a[i]) * grad_output[i];
    }
}

std::vector<Tensor> sigmoid_backward_impl_cpu(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), sigmoid_backward_impl_cpu, a, grad_output, grad_a);
    return {grad_a};
}

template <typename T>
static void softplus_backward_impl_cpu(TensorInfo<T> a, double beta, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = softplus_backward(a[i], T(beta)) * grad_output[i];
    }
}

std::vector<Tensor> softplus_backward_impl_cpu(Tensor a, double beta, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), softplus_backward_impl_cpu, a, beta, grad_output, grad_a);
    return {grad_a};
}

std::vector<Tensor> prod_backward_impl_cpu(Tensor a, int64_t dim, Tensor grad_output)
{
    throw std::runtime_error("not implemented");
    return {};
}

std::vector<Tensor> min_backward_impl_cpu(Tensor grad_output)
{
    Tensor grad_a = empty_like(grad_output);
    Tensor grad_b = empty_like(grad_output);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl_cpu, grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}

std::vector<Tensor> max_backward_impl_cpu(Tensor grad_output)
{
    Tensor grad_a = empty_like(grad_output);
    Tensor grad_b = empty_like(grad_output);
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl_cpu, grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}


// ================================================================================
// Tensor Create operators


Tensor empty(const SizeType& sizes, TensorOptions options)
{
    Tensor t(std::make_shared<TensorImpl>(sizes, options));
    return t;
}


template <typename T>
static void full_impl_cpu(TensorInfo<T> t, float value)
{
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t[i] = T(value);
    }
}

Tensor full(const SizeType& sizes, float value, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    SWITCH_MACRO_ALL(t.scalar_type(), full_impl_cpu, t, value);
    return t;
}


Tensor ones(const SizeType& sizes, TensorOptions options)
{
    return full(sizes, 1, options);
}


Tensor zeros(const SizeType& sizes, TensorOptions options)
{
    return full(sizes, 0, options);
}


template <typename T>
static void rand_float_impl_cpu(TensorInfo<T> t, std::mt19937& mersenne_engine)
{
    std::uniform_real_distribution<float> dist{0.f, 1.f};
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t[i] = T(dist(mersenne_engine));
    }
}

Tensor rand(const SizeType& sizes, TensorOptions options)
{
    static std::mt19937 mersenne_engine{572547235};
    
    Tensor t = empty(sizes, options);
    SWITCH_MACRO_ALL(t.scalar_type(), rand_float_impl_cpu, t, mersenne_engine);
    return t;
}

Tensor randint(int low, int high, const SizeType& sizes, TensorOptions options)
{
    static std::mt19937 mersenne_engine{572547235};
    std::uniform_int_distribution<int> dist{low, high};

    Tensor t = empty(sizes, options);
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t.data_ptr<int>()[i] = dist(mersenne_engine);
    }
    return t;
}

Tensor ones_like(Tensor t)
{
    return full_like(t, 1);
}
Tensor full_like(Tensor t, float value)
{
    Tensor t2 = empty_like(t);
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t2.data_ptr<float>()[i] = value;
    }
    return t2;
}

Tensor empty_like(Tensor t)
{
    Tensor t2(std::make_shared<TensorImpl>(t.sizes(), t.options()));
    return t2;
}
Tensor zeros_like(Tensor t)
{
    Tensor t2 = empty_like(t);
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t2.data_ptr<float>()[i] = 0;
    }
    return t2;
}
Tensor rand_like(Tensor t)
{
    static std::mt19937 mersenne_engine{572547235};
    std::uniform_real_distribution<float> dist{0.f, 1.f};

    Tensor t2 = empty_like(t);
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t2.data_ptr<float>()[i] = dist(mersenne_engine);
    }
    return t2;
}

template <typename T>
void print_impl_cpu(std::ostream& strm, TensorInfo<T> a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        std::cout << a[i] << " ";
    }
}

std::ostream& operator<<(std::ostream& strm, Tensor t)
{
    print_impl_cpu<float>(strm, t);
    return strm;
}

Tensor operator+=(Tensor a, Tensor b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, a);
    return a;
}

Tensor operator+=(Tensor a, double b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, a);
    return a;
}

Tensor operator-=(Tensor a, Tensor b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, a);
    return a;
}

Tensor operator-=(Tensor a, double b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, a);
    return a;
}

Tensor operator*=(Tensor a, Tensor b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, a);
    return a;
}

Tensor operator*=(Tensor a, double b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, a);
    return a;
}

Tensor operator/=(Tensor a, Tensor b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, a);
    return a;
}

Tensor operator/=(Tensor a, double b)
{
    assert(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, a);
    return a;
}



}  // namespace TINY_TORCH_NAMESPACE
