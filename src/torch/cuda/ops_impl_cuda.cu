#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
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
__launch_bounds__(128) static __global__ void range_impl(TensorInfo<T> a, double start, double end, double step)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(start) + T(i) * T(step);
}

void range_impl(Tensor a, double start, double end, double step)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), range_impl, a, start, end, step);
}

template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfo<T> a, double value)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(value);
}
template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfo<T> a, TensorInfo<T> value)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(value[0]);
}
template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfo<T> a, TensorInfo<T> values, int dim)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    auto index_a = a.LinearIndexToDimIndex(i);
    int d        = index_a[dim];
    a[index_a]   = values[d];
}
void fill_impl(Tensor& a, double value)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor value)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor values, int dim)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), fill_impl, a, values, dim);
}

template <typename TSource, typename TTarget>
__launch_bounds__(128) static __global__ void copy_and_convert_impl(TensorInfo<TSource> a, TensorInfo<TTarget> b)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    b[i] = TTarget(a[i]);
}


void copy_and_convert_impl(Tensor src, Tensor& target)
{
    CHECK_EQ(src.numel(), target.numel());
    switch (target.dtype())
    {
        case kInt32:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int32_t, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        case kInt64:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int64_t, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        case kFloat32:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), float, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        case kFloat64:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), double, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        default:
            throw std::runtime_error("invalid type");
    }
}

template <typename T>
__launch_bounds__(128) static __global__ void rand_float_impl(TensorInfo<T> a, float low, float high, uint64_t seed)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    curandState state;
    curand_init(seed, i, 0, &state);
    a[i] = T(curand_uniform(&state) * (high - low) + low);
}

void uniform_impl(Tensor& a, double mi, double ma)
{
    static int64_t seed = get_seed();
    srand((uint32_t)seed);
    seed = ::rand();
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), rand_float_impl, a, (float)mi, (float)ma, seed);
}

template <typename T>
__launch_bounds__(128) static __global__ void rand_int_impl(TensorInfo<T> a, int low, int high, uint64_t seed)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    curandState state;
    curand_init(seed, i, 0, &state);
    a[i] = T(curand(&state) % (high - low) + low);
}

void uniform_int_impl(Tensor& a, int low, int high)
{
    static int64_t seed = get_seed();
    srand((uint32_t)seed);
    seed = ::rand();
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), rand_int_impl, a, low, high, seed);
}


template <typename T>
__launch_bounds__(128) static __global__ void sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    atomicAdd(&result[0], a[i]);
}

void sum_impl(Tensor a, Tensor& result)
{
    auto stream = cuda::getCurrentCUDAStream();
    switch (a.scalar_type())
    {
        CUDA_CASE_MACRO(sum_impl<int32_t>, kInt32, a.numel(), a, result)
        CUDA_CASE_MACRO(sum_impl<float>, kFloat, a.numel(), a, result)
        CUDA_CASE_MACRO(sum_impl<double>, kDouble, a.numel(), a, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}

template <typename T>
__launch_bounds__(128) static __global__ void sum_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    int64_t linear_index_input = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (linear_index_input >= input.numel()) return;

    auto index_input  = input.LinearIndexToDimIndex(linear_index_input);
    auto result_index = index_input;
    result_index[dim] = 0;
    atomicAdd(&result[result_index], input[index_input]);
}

void sum_impl(Tensor a, int64_t dim, Tensor& result)
{
    auto stream = cuda::getCurrentCUDAStream();
    switch (a.scalar_type())
    {
        CUDA_CASE_MACRO(sum_impl<int32_t>, kInt32, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<float>, kFloat, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<double>, kDouble, a.numel(), a, dim, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}


template <typename T>
__launch_bounds__(128) static __global__ void min_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MIN(a[i], b[i]);
}

void min_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), min_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void max_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MAX(a[i], b[i]);
}


template <typename T>
__launch_bounds__(128) static __global__ void max_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;
    // atomicMax(&result[i], a[i]);
}
void max_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), max_impl, a, b, result);
}
void max_impl(Tensor a, Tensor& result)
{
    CHECK(false);
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), max_impl, a, result);
}
template <typename T, typename TIndex>
__launch_bounds__(128) static __global__
    void index_select_impl(TensorInfo<T> input, int64_t dim, TensorInfo<TIndex> index, TensorInfo<T> result)
{
    int64_t result_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (result_linear_index >= result.numel()) return;

    auto index_result    = result.LinearIndexToDimIndex(result_linear_index);
    auto index_input     = index_result;
    index_input[dim]     = index[index_result[dim]];
    result[index_result] = input[index_input];
}

template <typename TIndex>
static void index_select_helper(Tensor input, int64_t dim, TensorInfo<TIndex> index, Tensor result)
{
    CUDA_SWITCH_MACRO_ALL(result.scalar_type(), result.numel(), index_select_impl, input, dim, index, result);
}

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor& result)
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
__launch_bounds__(128) static __global__
    void index_add_impl(int64_t dim, TensorInfo<TIndex> index, TensorInfo<T> data, TensorInfo<T> result)
{
    int64_t input_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (input_linear_index >= data.numel()) return;

    auto index_input  = result.LinearIndexToDimIndex(input_linear_index);
    auto index_result = index_input;
    index_result[dim] = index[index_input[dim]];
    atomicAdd(&result[index_result], data[index_input]);
}

template <typename TIndex>
static void index_add_helper(Tensor data, int64_t dim, TensorInfo<TIndex> index, Tensor result)
{
    CUDA_SWITCH_MACRO_FLOAT(result.scalar_type(), result.numel(), index_add_impl, dim, index, data, result);
}

void index_add_impl(int64_t dim, Tensor index, Tensor data, Tensor& result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_add_helper<int32_t>(data, dim, index, result);
            break;
        case kLong:
            index_add_helper<int64_t>(data, dim, index, result);
            break;
    }
}



}  // namespace cuda_impl
}  // namespace tinytorch