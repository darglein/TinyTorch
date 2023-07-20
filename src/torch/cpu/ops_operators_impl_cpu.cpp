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

static std::pair<int64_t, int64_t> calculate_offsets(int64_t linearId, int64_t dims,
    int64_t* a_sizes, int64_t* b_sizes, int64_t* a_strides, int64_t* b_strides) 
{
    // This handles the case that if one tensor has size 1 along a dimension, the respective value is duplicated along
    // this dimension.

    int64_t offset_a = 0;
    int64_t offset_b = 0;

    for (int64_t i = dims - 1; i > 0; --i)
    {
        int64_t sa    = a_sizes[i];
        int64_t sb    = b_sizes[i];
        int64_t max_s = std::max(sa, sb);

        offset_a += (sa == 1) ? 0 : ((linearId % sa) * a_strides[i]);
        offset_b += (sb == 1) ? 0 : ((linearId % sb) * b_strides[i]);
        linearId /= max_s;
    }

    int64_t sa = a_sizes[0];
    int64_t sb = b_sizes[0];

    offset_a += (sa == 1) ? 0 : ((linearId % sa) * a_strides[0]);
    offset_b += (sb == 1) ? 0 : ((linearId % sb) * b_strides[0]);

    return {offset_a, offset_b};
}


template <typename T>
static void add_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t dims = result.dims;

    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto [offset_a, offset_b] = calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides);
        result[i]                 = a.data[offset_a] + b.data[offset_b];
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
    int64_t dims = result.dims;

    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto [offset_a, offset_b] = calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides);
        result[i] = a.data[offset_a] - b.data[offset_b];
    }
}

void sub_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, result);
}

template <typename T>
static void mult_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t dims = result.dims;

    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto [offset_a, offset_b] = calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides);
        result[i]                 = a.data[offset_a] * b.data[offset_b];
    }
}

void mult_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, result);
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
    int64_t dims = result.dims;

    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto [offset_a, offset_b] = calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides);
        result[i]                 = a.data[offset_a] / b.data[offset_b];
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
static void print_impl_cpu(std::ostream& strm, TensorInfo<T> a)
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
static void equal_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
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
    SWITCH_MACRO_ALL(a.scalar_type(), equal_impl_cpu, a, b, result);
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
