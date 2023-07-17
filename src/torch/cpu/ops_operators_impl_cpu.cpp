/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/ops_operators.h"
#include "torch/core/tensor_info.h"
#include "torch/cpu/ops_impl_cpu.h"



namespace tinytorch
{


template <typename T>
static void add_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] + b[i];
    }
}

void add_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, result);
}

template <typename T>
static void add_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] + b);
    }
}

void add_impl_cpu(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, result);
}

template <typename T>
static void sub_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] - b[i];
    }
}

void sub_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, result);
}


template <typename T>
static void mult_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    // This handles the case that if one tensor has size 1 along a dimension, the respective value is duplicated along
    // this dimension.

    int64_t dims = result.dims;

    for (int64_t i = 0; i < result.numel(); ++i)
    {
        int64_t linearId = i;
        int64_t offset_a = 0;
        int64_t offset_b = 0;

        for (int64_t i = dims - 1; i > 0; --i)
        {
            int64_t sa    = a.sizes[i];
            int64_t sb    = b.sizes[i];
            int64_t max_s = std::max(sa, sb);

            offset_a += (sa == 0) ? 0 : ((linearId % sa) * a.strides[i]);
            offset_b += (sb == 0) ? 0 : ((linearId % sb) * b.strides[i]);
            linearId /= max_s;
        }

        offset_a += linearId * a.strides[0];
        offset_b += linearId * b.strides[0];

        result[i] = a.data[offset_a] * b[offset_b];
    }
}

void mult_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    // Tensor result = empty(max_size(a, b), a.options());
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, result);
    // return result;
}

template <typename T>
static void mult_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] * b);
    }
}

void mult_impl_cpu(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, result);
}


template <typename T>
static void div_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = a[i] / b[i];
    }
}

void div_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, result);
}

template <typename T>
static void div_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] / b);
    }
}


template <typename T>
static void div_impl_cpu(double a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < b.numel(); ++i)
    {
        result[i] = T(a / b[i]);
    }
}
void div_impl_cpu(double a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(b.scalar_type(), div_impl_cpu, a, b, result);
}


template <typename T>
void print_impl_cpu(std::ostream& strm, TensorInfo<T> a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        strm << a[i] << " ";
    }
}
void print_impl_cpu(std::ostream& strm, Tensor t)
{
    print_impl_cpu<float>(strm, t);
}

template <typename T>
static void equals_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] == b);
    }
}
template <typename T>
static void less_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] < b);
    }
}
template <typename T>
static void greater_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i] > b);
    }
}

void equal_impl_cpu(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), equals_impl_cpu, a, b, result);
}
void less_impl_cpu(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), less_impl_cpu, a, b, result);
}
void greater_impl_cpu(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), greater_impl_cpu, a, b, result);
}

}  // namespace tinytorch
