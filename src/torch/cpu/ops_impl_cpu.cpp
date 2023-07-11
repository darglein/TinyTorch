/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/cpu/ops_impl_cpu.h"

#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"


namespace tinytorch
{

#define CASE_MACRO(func, type, scalar_type, ...) \
    case scalar_type:                            \
        func<type>(__VA_ARGS__);                 \
        break;

#define SWITCH_MACRO_FLOAT(real_scalar_type, func, ...)                \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, float, kFloat, __VA_ARGS__)                   \
        CASE_MACRO(func, double, kDouble, __VA_ARGS__)                 \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }

// TODO: Half!
#define SWITCH_MACRO_ALL(real_scalar_type, func, ...)                  \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, uint8_t, kUInt8, __VA_ARGS__)                 \
        CASE_MACRO(func, int16_t, kInt16, __VA_ARGS__)                 \
        CASE_MACRO(func, int32_t, kInt32, __VA_ARGS__)                 \
        CASE_MACRO(func, int64_t, kLong, __VA_ARGS__)                  \
        CASE_MACRO(func, float, kFloat, __VA_ARGS__)                   \
        CASE_MACRO(func, double, kDouble, __VA_ARGS__)                 \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
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
    SWITCH_MACRO_ALL(a.scalar_type(), fill_impl_cpu, a, value);
}


template <typename T>
static void range_impl_cpu(TensorInfo<T> a, double start, double end, double step)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = T(start) + T(i) * T(step);
    }
}
void range_impl_cpu(Tensor a, double start, double end, double step)
{
    SWITCH_MACRO_ALL(a.scalar_type(), range_impl_cpu, a, start, end, step);
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
template <typename T>
static void rand_int_impl_cpu(TensorInfo<T> t, std::mt19937& mersenne_engine, int low, int high)
{
    std::uniform_int_distribution<int> dist{low, high};
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t[i] = T(dist(mersenne_engine));
    }
}

void uniform(Tensor& t)
{
    static std::mt19937 mersenne_engine{572547235};
    SWITCH_MACRO_ALL(t.scalar_type(), rand_float_impl_cpu, t, mersenne_engine);
}
void uniform_int(Tensor& t, int low, int high)
{
    static std::mt19937 mersenne_engine{572547235};
    SWITCH_MACRO_ALL(t.scalar_type(), rand_int_impl_cpu, t, mersenne_engine, low, high);
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


template <typename T>
static void div_impl_cpu(double a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < b.numel(); ++i)
    {
        result[i] = T(a / b[i]);
    }
}
Tensor div_impl_cpu(double a, Tensor b)
{
    Tensor result = empty_like(b);
    SWITCH_MACRO_ALL(b.scalar_type(), div_impl_cpu, a, b, result);
    return result;
}

Tensor div_impl_cpu(Tensor a, double b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, result);
    return result;
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
    Tensor result = zeros({1}, a.options());
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
        T v       = a[i];
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
static void prod_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    int64_t to_prod = input.sizes[dim];
    int64_t count   = input.numel() / input.sizes[dim];
    CHECK_EQ(count, result.numel());

    for (int64_t c = 0; c < count; ++c)
    {
        int64_t linearId = c;

        int64_t input_offset = 0;
        for (int64_t i = dims - 1; i > 0; --i)
        {
            if (i != dim)
            {
                int64_t curDimIndex = linearId % input.sizes[i];
                input_offset += curDimIndex * input.strides[i];
                linearId /= input.sizes[i];
            }
        }

        if (dim != 0)
        {
            input_offset += linearId * input.strides[0];
        }

        T prod = T(1);

        for (int64_t p = 0; p < to_prod; ++p)
        {
            prod *= input[input_offset];
            input_offset += input.strides[dim];
        }

        result[c] = prod;
    }
}

Tensor prod_impl_cpu(Tensor input, int64_t dim)
{
    CHECK_LT(dim, input.dim());

    auto result_size = input.sizes();
    result_size[dim] = 1;

    Tensor result = empty(result_size, input.options());

    SWITCH_MACRO_ALL(input.scalar_type(), prod_impl_cpu, input, dim, result);

    result = result.squeeze(dim);

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
static void min_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    result[0] = std::numeric_limits<T>::max();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = std::min(a[i], result[0]);
    }
}

Tensor min_impl_cpu(Tensor a)
{
    Tensor result = empty({1}, a.options());
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl_cpu, a, result);
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
static void minmax_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<int64_t> indices, TensorInfo<T> result, bool calc_min)
{
    int64_t dims = input.dims;

    int64_t size_along_dim = input.sizes[dim];
    int64_t stride_along_dim = input.strides[dim];

    for (int64_t i = 0; i < result.numel(); ++i)
    {
        int64_t linearId = i;
        int64_t offset   = 0;
        for (int64_t i = dims - 1; i > 0; --i)
        {
            if (i != dim)
            {
                int64_t curDimIndex  = linearId % input.sizes[i];
                int64_t curDimOffset = curDimIndex * input.strides[i];
                offset += curDimOffset;
                linearId /= input.sizes[i];
            }
        }
        if (0 != dim)
        {
            offset += linearId * input.strides[0];
        }

        int64_t minmax_index = 0;
        T minmax_value       = calc_min ? std::numeric_limits<T>::max() : std::numeric_limits<T>::lowest();
        for (int64_t d = 0; d < size_along_dim; ++d)
        {
            T value = input.data[offset];
            bool cmp = calc_min ? (value < minmax_value) : (value > minmax_value);
            if (cmp)
            {
                minmax_value = value;
                minmax_index = d;
            }
            
            offset += stride_along_dim;
        }

        indices[i] = minmax_index;
        result[i]  = minmax_value;
    }
}

std::pair<Tensor, Tensor> min_impl_cpu(Tensor input, int64_t dim, bool keepdim)
{
    auto result_size = input.sizes();
    result_size[dim] = 1;

    Tensor result  = empty(result_size, input.options());
    Tensor indices = empty(result_size, input.options().dtype(kLong));

    SWITCH_MACRO_ALL(input.scalar_type(), minmax_impl_cpu, input, dim, indices, result, true);

    if (!keepdim)
    {
        result = result.squeeze(dim);
        indices = indices.squeeze(dim);
    }

    return {result, indices};
}

std::pair<Tensor, Tensor> max_impl_cpu(Tensor input, int64_t dim, bool keepdim)
{
    auto result_size = input.sizes();
    result_size[dim] = 1;

    Tensor result  = empty(result_size, input.options());
    Tensor indices = empty(result_size, input.options().dtype(kLong));

    SWITCH_MACRO_ALL(input.scalar_type(), minmax_impl_cpu, input, dim, indices, result, false);

    if (!keepdim)
    {
        result  = result.squeeze(dim);
        indices = indices.squeeze(dim);
    }

    return {result, indices};
}

template <typename T>
static void std_impl_cpu(TensorInfo<T> a, double mean, TensorInfo<T> result)
{
    T s = T(0);
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        T v = a[i] - T(mean);
        s += v * v;
    }
    result[0] = std::sqrt(s / a.numel());
}

Tensor std_impl_cpu(Tensor a)
{
    double mean   = (sum_impl_cpu(a) / (double)a.numel()).toDouble();
    Tensor result = empty({1}, a.options());
    SWITCH_MACRO_FLOAT(a.scalar_type(), std_impl_cpu, a, mean, result);
    return result;
}

template <typename T>
static void max_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    result[0] = std::numeric_limits<T>::min();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = std::max(a[i], result[0]);
    }
}

Tensor max_impl_cpu(Tensor a)
{
    Tensor result = empty({1}, a.options());
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl_cpu, a, result);
    return result;
}

template <typename T>
static void index_select_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<int64_t> index, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    int64_t to_copy = input.numel() / input.sizes[dim];
    for (int64_t index_index = 0; index_index < index.numel(); ++index_index)
    {
        int64_t slice        = index[index_index];
        int64_t input_start  = slice * input.strides[dim];
        int64_t result_start = index_index * result.strides[dim];

        for (int64_t c = 0; c < to_copy; ++c)
        {
            int64_t linearId = c;

            int64_t input_offset  = input_start;
            int64_t result_offset = result_start;
            for (int64_t i = dims - 1; i > 0; --i)
            {
                if (i != dim)
                {
                    int64_t curDimIndex = linearId % input.sizes[i];
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
    CHECK_LT(dim, input.dim());
    CHECK_EQ(index.dtype(), kLong);

    auto numel = index.numel();

    auto result_size = input.sizes().vec();
    result_size[dim] = numel;

    Tensor result = empty(result_size, input.options());
    SWITCH_MACRO_ALL(input.scalar_type(), index_select_impl_cpu, input, dim, index, result);
    return result;
}

template <typename T>
static void index_add_impl_cpu(int64_t dim, TensorInfo<int64_t> index, TensorInfo<T> data, TensorInfo<T> result)
{
    int64_t dims = result.dims;

    int64_t to_add = data.numel() / data.sizes[0];

    for (int64_t index_index = 0; index_index < index.numel(); ++index_index)
    {
        int64_t slice        = index[index_index];
        int64_t data_start   = index_index * data.strides[dim];
        int64_t result_start = slice * result.strides[dim];

        for (int64_t c = 0; c < to_add; ++c)
        {
            int64_t linearId = c;

            int64_t result_offset = result_start;
            for (int64_t i = dims - 1; i > 0; --i)
            {
                if (i != dim)
                {
                    int64_t curDimIndex = linearId % result.sizes[i];
                    result_offset += curDimIndex * result.strides[i];
                    linearId /= result.sizes[i];
                }
            }

            if (dim != 0)
            {
                result_offset += linearId * result.strides[0];
            }

            result.data[result_offset] += data[data_start + c];
        }
    }
}

Tensor index_add_impl_cpu(Tensor input, int64_t dim, Tensor index, Tensor data)
{
    CHECK_LT(dim, input.dim());
    CHECK_EQ(index.dtype(), kLong);
    CHECK_EQ(input.dim(), data.dim());
    CHECK_EQ(index.dim(), 1);
    CHECK_EQ(index.numel(), data.size(0));

    Tensor result = input.clone();
    SWITCH_MACRO_ALL(input.scalar_type(), index_add_impl_cpu, dim, index, data, result);
    return result;
}

template <typename T>
static void repeat_interleave_impl_cpu(TensorInfo<T> input, int64_t count, TensorInfo<T> result)
{
    int64_t to_copy = input.numel() / input.sizes[0];
    int64_t input_start = 0;
    int64_t output_start = 0;
    for (int64_t i = 0; i < input.sizes[0]; ++i)
    {
        for (int64_t c = 0; c < count; ++c)
        {
            for (int64_t j = 0; j < to_copy; ++j)
            {
                result[output_start + j] = input[input_start + j];
            }

            output_start += to_copy;
        }

        input_start += to_copy;
    }

    CHECK_EQ(input_start, input.numel());
    CHECK_EQ(output_start, result.numel());
}

Tensor repeat_interleave_impl_cpu(Tensor input, int64_t count)
{
    SizeType new_sizes = input.sizes();
    new_sizes[0] *= count;
    Tensor result = empty(new_sizes, input.options());
    SWITCH_MACRO_ALL(input.scalar_type(), repeat_interleave_impl_cpu, input, count, result);
    return result;
}

template <typename T>
static void stack_impl_cpu(TensorInfo<T> input, int64_t result_offset, TensorInfo<T> result)
{
    for (int64_t i = 0; i < input.numel(); ++i)
    {
        result[result_offset + i] = input[i];
    }
}

Tensor stack_impl_cpu(const std::vector<Tensor>& tensors)
{
    if (tensors.empty())
    {
        return {};
    }

    for (const auto& t : tensors)
    {
        CHECK_EQ(tensors.front().sizes(), t.sizes());
        CHECK_EQ(tensors.front().device(), t.device());
        CHECK_EQ(tensors.front().scalar_type(), t.scalar_type());
    }

    SizeType new_sizes = tensors.front().sizes();
    new_sizes.vec().insert(new_sizes.vec().begin(), tensors.size());

    int64_t individual_numel = tensors.front().numel();

    Tensor result = empty(new_sizes, tensors.front().options());

    int64_t offset = 0;
    for (const auto& t : tensors)
    {
        SWITCH_MACRO_ALL(t.scalar_type(), stack_impl_cpu, t, offset, result);
        offset += individual_numel;
    }

    return result;
}

template <typename T>
static void transpose_impl_cpu(TensorInfo<T> input, int64_t dim0, int64_t dim1, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    for (int64_t n = 0; n < input.numel(); ++n)
    {
        int64_t linearId = n;
        int64_t input_offset   = 0;
        int64_t output_offset   = 0;

        for (int64_t i = dims - 1; i > 0; --i)
        {
            int64_t curDimIndex = linearId % input.sizes[i];
            input_offset += curDimIndex * input.strides[i];

            int64_t j = (i == dim0) ? dim1 : (i == dim1) ? dim0 : i;
            output_offset += curDimIndex * result.strides[j];
            linearId /= input.sizes[i];
        }

        input_offset += linearId * input.strides[0];

        int64_t j = (0 == dim0) ? dim1 : (0 == dim1) ? dim0 : 0;
        output_offset += linearId * result.strides[j];


        result.data[output_offset] = input.data[input_offset];
    }
}

Tensor transpose_impl_cpu(Tensor input, int64_t dim0, int64_t dim1) 
{
    SizeType new_sizes = input.sizes();
    std::swap(new_sizes[dim0], new_sizes[dim1]);

    Tensor result = empty(new_sizes, input.options());
    SWITCH_MACRO_ALL(input.scalar_type(), transpose_impl_cpu, input, dim0, dim1, result);
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
        auto g        = grad_output[i];
        auto [ga, gb] = div_backward(a[i], b[i]);
        grad_a[i]     = ga * g;
        grad_b[i]     = gb * g;
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
        auto g        = grad_output[i];
        auto [ga, gb] = div_backward(a[i], T(b));
        grad_a[i]     = ga * g;
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
        auto g        = grad_output[i];
        auto [ga, gb] = div_backward(T(a), b[i]);
        grad_b[i]     = gb * g;
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
    CHECK_EQ(grad_output.numel(), 1);
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
template <typename T>
void print_impl_cpu(std::ostream& strm, TensorInfo<T> a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        strm << a[i] << " ";
    }
}

std::ostream& operator<<(std::ostream& strm, Tensor t)
{
    print_impl_cpu<float>(strm, t);
    return strm;
}

Tensor operator+=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, a);
    return a;
}

Tensor operator+=(Tensor a, double b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), add_impl_cpu, a, b, a);
    return a;
}

Tensor operator-=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, a);
    return a;
}

Tensor operator-=(Tensor a, double b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), sub_impl_cpu, a, b, a);
    return a;
}

Tensor operator*=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, a);
    return a;
}

Tensor operator*=(Tensor a, double b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), mult_impl_cpu, a, b, a);
    return a;
}

Tensor operator/=(Tensor a, Tensor b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, a);
    return a;
}

Tensor operator/=(Tensor a, double b)
{
    CHECK(!a.requires_grad());
    SWITCH_MACRO_ALL(a.scalar_type(), div_impl_cpu, a, b, a);
    return a;
}

Tensor operator==(Tensor a, double b)
{
    CHECK(!a.requires_grad());
    Tensor t2 = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), equals_impl_cpu, a, b, t2);
    return t2;
}


template <typename T>
static void to_double_cpu(TensorInfo<T> a, TensorInfo<double> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = double(a[i]);
    }
}

template <typename T>
static void from_double_cpu(TensorInfo<double> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(a[i]);
    }
}

Tensor to(Tensor a, ScalarType other_type)
{
    Tensor t2 = empty_like(a, TensorOptions().dtype(kDouble));
    SWITCH_MACRO_ALL(a.scalar_type(), to_double_cpu, a, t2);

    Tensor result = empty_like(a, TensorOptions().dtype(other_type));
    SWITCH_MACRO_ALL(result.scalar_type(), from_double_cpu, t2, result);
    return result;
}

template <typename T>
static void copy_cpu(TensorInfo<T> src, TensorInfo<T> dst)
{
    for (int64_t i = 0; i < src.numel(); ++i)
    {
        dst[i] = src[i];
    }
}

void copy(Tensor src, Tensor target)
{
    CHECK_EQ(src.numel(), target.numel());
    SWITCH_MACRO_ALL(src.scalar_type(), copy_cpu, src, target);
}


}  // namespace tinytorch
