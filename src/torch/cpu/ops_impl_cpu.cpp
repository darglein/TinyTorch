/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/cpu/ops_impl_cpu.h"

#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/ops_operators.h"
#include "torch/core/tensor_info.h"



namespace tinytorch
{
namespace cpu_impl
{

void to_impl_cpu_cuda(Tensor a, Tensor b)
{
#ifdef TT_HAS_CUDA
    CHECK(a.is_contiguous());
    CHECK(b.is_contiguous());
    int64_t bytes = a.element_size() * a.numel();
    auto type     = (b.device() == kCPU) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    cudaMemcpy(b.data_ptr(), a.data_ptr(), bytes, type);
#else
    CHECK(false);
#endif
}

template <typename T>
static void fill_impl(TensorInfo<T> a, double value)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = T(value);
    }
}
template <typename T>
static void fill_impl(TensorInfo<T> a, TensorInfo<T> value)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = value[0];
    }
}
template <typename T>
static void fill_impl(TensorInfo<T> a, TensorInfo<T> values, int dim)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto index_a = a.LinearIndexToDimIndex(i);
        int d        = index_a[dim];
        a[index_a]   = values[d];
    }
}

void fill_impl(Tensor& a, double value)
{
    SWITCH_MACRO_ALL(a.scalar_type(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor value)
{
    SWITCH_MACRO_ALL(a.scalar_type(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor values, int dim)
{
    SWITCH_MACRO_ALL(a.scalar_type(), fill_impl, a, values, dim);
}

template <typename T>
static void range_impl(TensorInfo<T> a, double start, double end, double step)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = T(start) + T(i) * T(step);
    }
}
void range_impl(Tensor a, double start, double end, double step)
{
    SWITCH_MACRO_ALL(a.scalar_type(), range_impl, a, start, end, step);
}


template <typename T>
static void rand_float_impl(TensorInfo<T> t, std::mt19937& mersenne_engine, float low, float high)
{
    std::uniform_real_distribution<float> dist{low, high};
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t[i] = T(dist(mersenne_engine));
    }
}

void uniform_impl(Tensor& t, double mi, double ma)
{
    static std::mt19937 mersenne_engine{(uint32_t)get_seed()};
    SWITCH_MACRO_ALL(t.scalar_type(), rand_float_impl, t, mersenne_engine, (float)mi, (float)ma);
}

template <typename T>
static void rand_int_impl(TensorInfo<T> t, std::mt19937& mersenne_engine, int low, int high)
{
    std::uniform_int_distribution<int> dist{low, high};
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t[i] = T(dist(mersenne_engine));
    }
}

void uniform_int_impl(Tensor& t, int low, int high)
{
    static std::mt19937 mersenne_engine{(uint32_t)get_seed()};
    SWITCH_MACRO_ALL(t.scalar_type(), rand_int_impl, t, mersenne_engine, low, high);
}



template <typename T>
static void sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] += a[i];
    }
}

void sum_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sum_impl, a, result);
}

template <typename T>
static void sum_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    for (int64_t linear_index_input = 0; linear_index_input < input.numel(); ++linear_index_input)
    {
        auto index_input  = input.LinearIndexToDimIndex(linear_index_input);
        auto result_index = index_input;
        result_index[dim] = 0;
        result[result_index] += input[index_input];
    }
}

void sum_impl(Tensor a, int64_t dim, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sum_impl, a, dim, result);
}


template <typename T>
static void prod_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    int64_t dim_size = input.sizes[dim];
    int64_t count    = input.numel() / input.sizes[dim];
    CHECK_EQ(count, result.numel());

    for (int64_t c = 0; c < count; ++c)
    {
        int64_t input_offset = index_along_dim(c, dims, dim, input.sizes, input.strides);

        T prod = T(1);

        for (int64_t p = 0; p < dim_size; ++p)
        {
            prod *= input[input_offset];
            input_offset += input.strides[dim];
        }

        result[c] = prod;
    }
}

void prod_impl(Tensor input, int64_t dim, Tensor& result)
{
    /*
    CHECK_LT(dim, input.dim());

    auto result_size = input.sizes();
    result_size[dim] = 1;

    Tensor result = empty(result_size, input.options());
    */

    SWITCH_MACRO_ALL(input.scalar_type(), prod_impl, input, dim, result);
    result = result.squeeze(dim);
}

template <typename T>
static void min_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::min(a[i], b[i]);
    }
}

void min_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl, a, b, result);
}

template <typename T>
static void min_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    result[0] = std::numeric_limits<T>::infinity();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = std::min(a[i], result[0]);
    }
}

void min_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl, a, result);
}

template <typename T>
static void max_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::max(a[i], b[i]);
    }
}

void max_impl(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl, a, b, result);
}

template <typename T>
static void max_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    result[0] = -std::numeric_limits<T>::infinity();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = std::max(a[i], result[0]);
    }
}

void max_impl(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl, a, result);
}

template <typename T>
static void minmax_impl(TensorInfo<T> input, int64_t dim, TensorInfo<int64_t> indices, TensorInfo<T> result,
                        bool calc_min)
{
    int64_t dims = input.dims;

    int64_t size_along_dim   = input.sizes[dim];
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
            T value  = input.data[offset];
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

void min_impl(Tensor input, int64_t dim, bool keepdim, Tensor& result, Tensor& indices)
{
    SWITCH_MACRO_ALL(input.scalar_type(), minmax_impl, input, dim, indices, result, true);

    if (!keepdim)
    {
        result  = result.squeeze(dim);
        indices = indices.squeeze(dim);
    }
}

void max_impl(Tensor input, int64_t dim, bool keepdim, Tensor& result, Tensor& indices)
{
    SWITCH_MACRO_ALL(input.scalar_type(), minmax_impl, input, dim, indices, result, false);

    if (!keepdim)
    {
        result  = result.squeeze(dim);
        indices = indices.squeeze(dim);
    }
}


template <typename T, typename Indextype>
static void index_select_impl(TensorInfo<T> input, int64_t dim, TensorInfo<Indextype> index, TensorInfo<T> result)
{
    for (int64_t result_linear_index = 0; result_linear_index < result.numel(); ++result_linear_index)

    {
        auto index_result = result.LinearIndexToDimIndex(result_linear_index);

        auto index_input = index_result;
        index_input[dim] = index[index_result[dim]];


        result[index_result] = input[index_input];
    }
}

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor& result)
{
    CHECK(input.is_cpu());
    CHECK_LT(dim, input.dim());


    switch (index.dtype())
    {
        case kInt32:
        {
            SWITCH_MACRO_ALL_DUAL(input.scalar_type(), int32_t, index_select_impl, input, dim, index, result);
            break;
        }
        case kInt64:
        {
            SWITCH_MACRO_ALL_DUAL(input.scalar_type(), int64_t, index_select_impl, input, dim, index, result);
            break;
        }
        default:
            throw std::runtime_error("invalid index type");
    }
}

template <typename T, typename TIndex>
static void index_add_impl(int64_t dim, TensorInfo<TIndex> index, TensorInfo<T> data, TensorInfo<T> result)
{
    for (int64_t input_linear_index = 0; input_linear_index < data.numel(); ++input_linear_index)
    {
        auto index_input = data.LinearIndexToDimIndex(input_linear_index);

        auto index_result = index_input;
        index_result[dim] = index[index_input[dim]];

        result[index_result] += data[index_input];
    }
}

template <typename TIndex>
static void index_add_helper(int64_t dim, TensorInfo<TIndex> index, Tensor data, Tensor result)
{
    SWITCH_MACRO_ALL(data.scalar_type(), index_add_impl, dim, index, data, result);
}

void index_add_impl(int64_t dim, Tensor index, Tensor data, Tensor& result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_add_helper<int32_t>(dim, index, data, result);
            break;
        case kLong:
            index_add_helper<int64_t>(dim, index, data, result);
            break;
        default:
            throw std::runtime_error("invalid type");
    }
}

template <typename T>
static void repeat_interleave_impl(TensorInfo<T> input, int64_t count, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto index_result = result.LinearIndexToDimIndex(i);
        auto index_input = input.LinearIndexToDimIndex(i / count);
        result[index_result] = input[index_input];
    }
}

void repeat_interleave_impl(Tensor input, int64_t count, Tensor& result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), repeat_interleave_impl, input, count, result);
}


template <typename T>
static void transpose_impl(TensorInfo<T> input, int64_t dim0, int64_t dim1, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto index_result = result.LinearIndexToDimIndex(i);
        auto index_input = index_result;
        std::swap(index_input[dim0], index_input[dim1]);
        result[index_result] = input[index_input];
    }
}

void transpose_impl(Tensor input, int64_t dim0, int64_t dim1, Tensor& result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), transpose_impl, input, dim0, dim1, result);
}

// ================================================================================================================



template <typename T>
static void sum_backward_impl(TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    auto g = grad_output[0];
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = g;
    }
}

void sum_backward_impl(Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_ALL(grad_output.scalar_type(), sum_backward_impl, grad_output, grad_a);
}


void prod_backward_impl(Tensor a, int64_t dim, Tensor grad_output, Tensor& grad_a)
{
    throw std::runtime_error("not implemented");
}

template <typename T>
static void one_backward_impl(TensorInfo<T> grad_output, TensorInfo<T> grad_a, TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < grad_output.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = g;
        grad_b[i] = g;
    }
}

void min_backward_impl(Tensor grad_output, Tensor& grad_a, Tensor& grad_b)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl, grad_output, grad_a, grad_b);
}

void max_backward_impl(Tensor grad_output, Tensor& grad_a, Tensor& grad_b)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl, grad_output, grad_a, grad_b);
}


template <typename TSource, typename TTarget>
static void copy_and_convert_impl_kernel(TensorInfo<TSource> src, TensorInfo<TTarget> target)
{
    for (int64_t i = 0; i < src.numel(); ++i)
    {
        target[i] = TTarget(src[i]);
    }
}

void copy_and_convert_impl(Tensor src, Tensor& target)
{
    CHECK_EQ(src.numel(), target.numel());
    switch (target.dtype())
    {
        case kInt32:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int32_t, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kInt64:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int64_t, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kFloat32:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), float, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kFloat64:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), double, copy_and_convert_impl_kernel, src, target);
            break;
        }
        default:
            throw std::runtime_error("invalid type");
    }
}

template <typename T>
static void clamp_impl_(TensorInfo<T> src, double low, double high)
{
    T low_t  = std::isfinite(low) ? T(low) : std::numeric_limits<T>::lowest();
    T high_t = std::isfinite(high) ? T(high) : std::numeric_limits<T>::max();

    for (int64_t i = 0; i < src.numel(); ++i)
    {
        src[i] = std::min<T>(high_t, std::max<T>(src[i], low_t));
    }
}
void clamp_impl_(Tensor& a, double low, double high)
{
    SWITCH_MACRO_ALL(a.scalar_type(), clamp_impl_, a, low, high);
}



}  // namespace cpu_impl
}  // namespace tinytorch