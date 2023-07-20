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
static void rand_float_impl_cpu(TensorInfo<T> t, std::mt19937& mersenne_engine, float low, float high)
{
    std::uniform_real_distribution<float> dist{low, high};
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t[i] = T(dist(mersenne_engine));
    }
}

void uniform_impl_cpu(Tensor& t, double mi, double ma)
{
    static std::mt19937 mersenne_engine{(uint32_t)get_seed()};
    SWITCH_MACRO_ALL(t.scalar_type(), rand_float_impl_cpu, t, mersenne_engine, (float)mi, (float)ma);
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

void uniform_int_impl_cpu(Tensor& t, int low, int high)
{
    static std::mt19937 mersenne_engine{(uint32_t)get_seed()};
    SWITCH_MACRO_ALL(t.scalar_type(), rand_int_impl_cpu, t, mersenne_engine, low, high);
}


template <typename T>
static void sqrt_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i]= T(std::sqrt(a[i]));
    }
}

void sqrt_impl_cpu(Tensor a, Tensor& result) {

    SWITCH_MACRO_ALL(a.scalar_type(), sqrt_impl_cpu, a, result);
}


template <typename T>
static void sum_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] += a[i];
    }
}

void sum_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sum_impl_cpu, a, result);
}

template <typename T>
static void sum_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    int64_t dim_size = input.sizes[dim];
    int64_t count   = input.numel() / input.sizes[dim];
    CHECK_EQ(count, result.numel());

    for (int64_t c = 0; c < count; ++c)
    {
        int64_t input_offset = index_along_dim(c, dims, dim, input.sizes, input.strides);

        T sum = T(0);

        for (int64_t p = 0; p < dim_size; ++p)
        {
            sum += input[input_offset];
            input_offset += input.strides[dim];
        }

        result[c] = sum;
    }
}

void sum_impl_cpu(Tensor a, int64_t dim, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sum_impl_cpu, a, dim, result);
}

template <typename T>
static void log_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::log(a[i]);
    }
}

void log_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), log_impl_cpu, a, result);
}

template <typename T>
static void log1p_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::log1p(a[i]);
    }
}

void log1p_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), log1p_impl_cpu, a, result);
}

template <typename T>
static void exp_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::exp(a[i]);
    }
}

void exp_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), exp_impl_cpu, a, result);
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

void sign_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sign_impl_cpu, a, result);
}

template <typename T>
static void pow_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = T(std::pow(a[i], b));
    }
}

void pow_impl_cpu(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), pow_impl_cpu, a, b, result);
}

template <typename T>
static void sin_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::sin(a[i]);
    }
}

void sin_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sin_impl_cpu, a, result);
}

template <typename T>
static void cos_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::cos(a[i]);
    }
}

void cos_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), cos_impl_cpu, a, result);
}

template <typename T>
static void relu_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = relu(a[i]);
    }
}

void relu_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), relu_impl_cpu, a, result);
}

template <typename T>
static void sigmoid_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = sigmoid(a[i]);
    }
}

void sigmoid_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sigmoid_impl_cpu, a, result);
}

template <typename T>
static void softplus_impl_cpu(TensorInfo<T> a, double beta, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = softplus(a[i], T(beta));
    }
}

void softplus_impl_cpu(Tensor a, double beta, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), softplus_impl_cpu, a, beta, result);
}

template <typename T>
static void prod_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    int64_t dim_size = input.sizes[dim];
    int64_t count   = input.numel() / input.sizes[dim];
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

void prod_impl_cpu(Tensor input, int64_t dim, Tensor& result)
{
    /*
    CHECK_LT(dim, input.dim());

    auto result_size = input.sizes();
    result_size[dim] = 1;

    Tensor result = empty(result_size, input.options());
    */

    SWITCH_MACRO_ALL(input.scalar_type(), prod_impl_cpu, input, dim, result);
    result = result.squeeze(dim);
}

template <typename T>
static void min_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::min(a[i], b[i]);
    }
}

void min_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl_cpu, a, b, result);
}

template <typename T>
static void min_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    result[0] = std::numeric_limits<T>::infinity();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = std::min(a[i], result[0]);
    }
}

void min_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl_cpu, a, result);
}

template <typename T>
static void max_impl_cpu(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::max(a[i], b[i]);
    }
}

void max_impl_cpu(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl_cpu, a, b, result);
}

template <typename T>
static void max_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    result[0] = -std::numeric_limits<T>::infinity();
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = std::max(a[i], result[0]);
    }
}

void max_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl_cpu, a, result);
}

template <typename T>
static void minmax_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<int64_t> indices, TensorInfo<T> result,
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

void min_impl_cpu(Tensor input, int64_t dim, bool keepdim, Tensor& result, Tensor& indices)
{
    SWITCH_MACRO_ALL(input.scalar_type(), minmax_impl_cpu, input, dim, indices, result, true);

    if (!keepdim)
    {
        result  = result.squeeze(dim);
        indices = indices.squeeze(dim);
    }
}

void max_impl_cpu(Tensor input, int64_t dim, bool keepdim, Tensor& result, Tensor& indices)
{
    SWITCH_MACRO_ALL(input.scalar_type(), minmax_impl_cpu, input, dim, indices, result, false);

    if (!keepdim)
    {
        result  = result.squeeze(dim);
        indices = indices.squeeze(dim);
    }
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

void std_impl_cpu(Tensor a, Tensor& result)
{
    double mean = a.mean().toDouble();
    SWITCH_MACRO_FLOAT(a.scalar_type(), std_impl_cpu, a, mean, result);
}

template <typename T>
static void abs_impl_cpu(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[i] = std::abs(a[i]);
    }
}

void abs_impl_cpu(Tensor a, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), abs_impl_cpu, a, result);
}

template <typename T, typename TIndex>
static void index_select_impl_cpu(TensorInfo<T> input, int64_t dim, TensorInfo<TIndex> index, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    int64_t slice_size = input.numel() / input.sizes[dim];
    for (int64_t index_index = 0; index_index < index.numel(); ++index_index)
    {
        int64_t slice        = index[index_index];
        int64_t input_start  = slice * input.strides[dim];
        int64_t result_start = index_index * result.strides[dim];

        for (int64_t c = 0; c < slice_size; ++c)
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

template <typename TIndex>
static void index_select_helper(Tensor input, int64_t dim, TensorInfo<TIndex> index, Tensor result) 
{
    SWITCH_MACRO_ALL(input.scalar_type(), index_select_impl_cpu, input, dim, index, result);
}

void index_select_impl_cpu(Tensor input, int64_t dim, Tensor index, Tensor& result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_select_helper<int32_t>(input, dim, index, result);
            break;
        case kLong:
            index_select_helper<int64_t>(input, dim, index, result);
            break;
    }
}

template <typename T, typename TIndex>
static void index_add_impl_cpu(int64_t dim, TensorInfo<TIndex> index, TensorInfo<T> data, TensorInfo<T> result)
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

template <typename TIndex>
static void index_add_helper(Tensor input, int64_t dim, TensorInfo<TIndex> index, Tensor data, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), index_add_impl_cpu, dim, index, data, result);
}

void index_add_impl_cpu(Tensor input, int64_t dim, Tensor index, Tensor data, Tensor& result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_add_helper<int32_t>(input, dim, index, data, result);
            break;
        case kLong:
            index_add_helper<int64_t>(input, dim, index, data, result);
            break;
    }
}

template <typename T>
static void repeat_interleave_impl_cpu(TensorInfo<T> input, int64_t count, TensorInfo<T> result)
{
    int64_t to_copy      = input.numel() / input.sizes[0];
    int64_t input_start  = 0;
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

void repeat_interleave_impl_cpu(Tensor input, int64_t count, Tensor& result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), repeat_interleave_impl_cpu, input, count, result);
}

template <typename T>
static void stack_impl_cpu(TensorInfo<T> input, int64_t result_offset, TensorInfo<T> result)
{
    for (int64_t i = 0; i < input.numel(); ++i)
    {
        result[result_offset + i] = input[i];
    }
}

void stack_impl_cpu(const std::vector<Tensor>& tensors, Tensor& result)
{
    int64_t individual_numel = tensors.front().numel();

    int64_t offset = 0;
    for (const auto& t : tensors)
    {
        SWITCH_MACRO_ALL(t.scalar_type(), stack_impl_cpu, t, offset, result);
        offset += individual_numel;
    }
}

template <typename T>
static void transpose_impl_cpu(TensorInfo<T> input, int64_t dim0, int64_t dim1, TensorInfo<T> result)
{
    int64_t dims = input.dims;

    for (int64_t n = 0; n < input.numel(); ++n)
    {
        int64_t linearId      = n;
        int64_t input_offset  = 0;
        int64_t output_offset = 0;

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

void transpose_impl_cpu(Tensor input, int64_t dim0, int64_t dim1, Tensor& result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), transpose_impl_cpu, input, dim0, dim1, result);
}

// ================================================================================================================


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

void sum_backward_impl_cpu(Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_ALL(grad_output.scalar_type(), sum_backward_impl_cpu, grad_output, grad_a);
}

template <typename T>
static void log_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = log_backward(a[i]) * grad_output[i];
    }
}

void log_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), log_backward_impl_cpu, a, grad_output, grad_a);
}

template <typename T>
static void log1p_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = log1p_backward(a[i]) * grad_output[i];
    }
}

void log1p_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), log1p_backward_impl_cpu, a, grad_output, grad_a);
}

template <typename T>
static void exp_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = std::exp(a[i]) * grad_output[i];
    }
}

void exp_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), exp_backward_impl_cpu, a, grad_output, grad_a);
}

void sign_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    throw std::runtime_error("not implemented");
}

template <typename T>
static void pow_backward_impl_cpu(TensorInfo<T> a, double b, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = pow_backward(a[i], T(b)) * grad_output[i];
    }
}

void pow_backward_impl_cpu(Tensor a, double b, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), pow_backward_impl_cpu, a, b, grad_output, grad_a);
}

template <typename T>
static void sin_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = sin_backward(a[i]) * grad_output[i];
    }
}

void sin_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), sin_backward_impl_cpu, a, grad_output, grad_a);
}

template <typename T>
static void cos_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = cos_backward(a[i]) * grad_output[i];
    }
}

void cos_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), cos_backward_impl_cpu, a, grad_output, grad_a);
}

template <typename T>
static void relu_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = relu_backward(a[i]) * grad_output[i];
    }
}

void relu_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), relu_backward_impl_cpu, a, grad_output, grad_a);
}

template <typename T>
static void sigmoid_backward_impl_cpu(TensorInfo<T> a, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = sigmoid_backward(a[i]) * grad_output[i];
    }
}

void sigmoid_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), sigmoid_backward_impl_cpu, a, grad_output, grad_a);
}

template <typename T>
static void softplus_backward_impl_cpu(TensorInfo<T> a, double beta, TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = softplus_backward(a[i], T(beta)) * grad_output[i];
    }
}

void softplus_backward_impl_cpu(Tensor a, double beta, Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), softplus_backward_impl_cpu, a, beta, grad_output, grad_a);
}

void prod_backward_impl_cpu(Tensor a, int64_t dim, Tensor grad_output, Tensor& grad_a)
{
    throw std::runtime_error("not implemented");
}

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

void min_backward_impl_cpu(Tensor grad_output, Tensor& grad_a, Tensor& grad_b)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl_cpu, grad_output, grad_a, grad_b);
}

void max_backward_impl_cpu(Tensor grad_output, Tensor& grad_a, Tensor& grad_b)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl_cpu, grad_output, grad_a, grad_b);
}

template <typename T>
static void to_double_impl_cpu(TensorInfo<T> a, TensorInfo<double> result)
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

Tensor to_impl_cpu(Tensor a, ScalarType other_type)
{
    Tensor t2 = empty_like(a, TensorOptions().dtype(kDouble));
    SWITCH_MACRO_ALL(a.scalar_type(), to_double_impl_cpu, a, t2);

    Tensor result = empty_like(a, TensorOptions().dtype(other_type));
    SWITCH_MACRO_ALL(result.scalar_type(), from_double_cpu, t2, result);
    return result;
}

template <typename TSource, typename TTarget>
static void copy_and_convert_impl_cpu(TensorInfo<TSource> src, TensorInfo<TTarget> target)
{
    for (int64_t i = 0; i < src.numel(); ++i)
    {
        target[i] = TTarget(src[i]);
    }
}

template <typename TSource>
static void copy_and_convert_helper_cpu(TensorInfo<TSource> src, Tensor target)
{
    switch (target.scalar_type())
    {
        case kUInt8:
            copy_and_convert_impl_cpu<TSource, uint8_t>(src, target);
            break;
        case kInt16:
            copy_and_convert_impl_cpu<TSource, int16_t>(src, target);
            break;
        case kInt32:
            copy_and_convert_impl_cpu<TSource, int32_t>(src, target);
            break;
        case kLong:
            copy_and_convert_impl_cpu<TSource, int64_t>(src, target);
            break;
        case kFloat:
            copy_and_convert_impl_cpu<TSource, float>(src, target);
            break;
        case kDouble:
            copy_and_convert_impl_cpu<TSource, double>(src, target);
            break;
        default:
            CHECK(false) << "invalid input type " << target.scalar_type();
    }
}

void copy_and_convert_impl_cpu(Tensor src, Tensor& target)
{
    CHECK_EQ(src.numel(), target.numel());
    SWITCH_MACRO_ALL(src.scalar_type(), copy_and_convert_helper_cpu, src, target);
}

template <typename T>
static void clamp_impl_cpu_(TensorInfo<T> src, double low, double high)
{
    T low_t  = std::isfinite(low) ? T(low) : std::numeric_limits<T>::lowest();
    T high_t = std::isfinite(high) ? T(high) : std::numeric_limits<T>::max();

    for (int64_t i = 0; i < src.numel(); ++i)
    {
        src[i] = std::min<T>(high_t, std::max<T>(src[i], low_t));
    }
}
void clamp_impl_cpu_(Tensor& a, double low, double high)
{
    SWITCH_MACRO_ALL(a.scalar_type(), clamp_impl_cpu_, a, low, high);
}



}  // namespace tinytorch
