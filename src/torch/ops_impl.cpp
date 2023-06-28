/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops.h"
#include "tensor.h"

#include "tensor_info.h"
namespace tinytorch
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
    }

template <typename T>
void fill_impl(TensorInfo<T> a, float value)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = value;
    }
}
void fill_impl(Tensor a, float value)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), fill_impl, a, value);
}

template <typename T>
void square_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto v    = a[i];
        result[i] = v * v;
    }
}

Tensor square_impl(Tensor a)
{
    Tensor result = empty_like(a);
    square_impl<float>(a, result);
    return result;
}

template <typename T>
void sub_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] - b[i];
    }
}

Tensor sub_impl(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    sub_impl<float>(a, b, result);
    return result;
}

template <typename T>
void add_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] + b[i];
    }
}

Tensor add_impl(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    add_impl<float>(a, b, result);
    return result;
}

template <typename T>
void mult_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] * b[i];
    }
}

Tensor mult_impl(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    mult_impl<float>(a, b, result);
    return result;
}


template <typename T>
void sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] += a[i];
    }
}
Tensor sum_impl(Tensor a)
{
    Tensor result = zeros({1});
    sum_impl<float>(a, result);
    return result;
}

// ================================================================================================================

template <typename T>
void square_backward_impl(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        grad_a[i] = 2 * a[i] * grad_output[i];
    }
}

std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    square_backward_impl<float>(a, grad_output, grad_a);
    return {grad_a};
}


template <typename T>
void square_backward_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> grad_output, TensorInfo<T> grad_a,
                          TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = b[i] * g;
        grad_b[i] = a[i] * g;
    }
}

std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output)
{
    Tensor grad_a = empty_like(a);
    Tensor grad_b = empty_like(b);
    square_backward_impl<float>(a, b, grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}


template <typename T>
void add_backward_impl(TensorInfo<T> grad_output, TensorInfo<T> grad_a, TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < grad_output.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = g;
        grad_b[i] = g;
    }
}


std::vector<Tensor> add_backward_impl(Tensor grad_output)
{
    Tensor grad_a = empty_like(grad_output);
    Tensor grad_b = empty_like(grad_output);
    add_backward_impl<float>(grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}


template <typename T>
void sub_backward_impl(TensorInfo<T> grad_output, TensorInfo<T> grad_a, TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < grad_output.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = g;
        grad_b[i] = -g;
    }
}


std::vector<Tensor> sub_backward_impl(Tensor grad_output)
{
    Tensor grad_a = empty_like(grad_output);
    Tensor grad_b = empty_like(grad_output);
    sub_backward_impl<float>(grad_output, grad_a, grad_b);
    return {grad_a, grad_b};
}

template <typename T>
void sum_backward_impl(TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        auto g    = grad_output[0];
        grad_a[i] = g;
    }
}


std::vector<Tensor> sum_backward_impl(const std::vector<int64_t>& input_sizes, Tensor grad_output)
{
    assert(grad_output.numel() == 1);
    Tensor grad_a = empty(input_sizes);
    sum_backward_impl<float>(grad_output, grad_a);
    return {grad_a};
}


// ================================================================================
// Tensor Create operators


Tensor empty(const std::vector<int64_t>& sizes, TensorOptions options)
{
    Tensor t(std::make_shared<TensorImpl>(sizes, options));
    return t;
}


Tensor full(const std::vector<int64_t>& sizes, float value, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t.data_ptr<float>()[i] = value;
    }
    return t;
}


Tensor ones(const std::vector<int64_t>& sizes, TensorOptions options)
{
    return full(sizes, 1, options);
}


Tensor zeros(const std::vector<int64_t>& sizes, TensorOptions options)
{
    return full(sizes, 0, options);
}


Tensor rand(const std::vector<int64_t>& sizes, TensorOptions options)
{
    static std::mt19937 mersenne_engine{572547235};
    std::uniform_real_distribution<float> dist{0.f, 1.f};

    Tensor t = empty(sizes, options);
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t.data_ptr<float>()[i] = dist(mersenne_engine);
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
void print_impl(std::ostream& strm, TensorInfo<T> a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        std::cout << a[i] << " ";
    }
}

std::ostream& operator<<(std::ostream& strm, Tensor t)
{
    print_impl<float>(strm, t);
    return strm;
}
Tensor& operator+=(Tensor& a, Tensor b)
{
    assert(!a.requires_grad());
    add_impl<float>(a, b, a);
    return a;
}

}  // namespace tinytorch
