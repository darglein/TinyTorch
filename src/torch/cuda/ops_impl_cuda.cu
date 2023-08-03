#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/atomic_minmax.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{
namespace cuda_impl
{

template <typename T>
__launch_bounds__(128) static __global__ void range_impl(TensorInfoCuda<T> a, double start, double end, double step)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(start + i * step);
}

void range_impl(Tensor a, double start, double end, double step)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), range_impl, a, start, end, step);
}

template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfoCuda<T> a, double value)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(value);
}
template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> value)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(value[0]);
}
template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> values, int dim)
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

template <typename T>
__launch_bounds__(128) static __global__
    void permute_impl(TensorInfo<T> src, TensorInfo<T> result, DimIndexStruct<25> new_dims)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= src.numel()) return;

    auto index_src    = src.LinearIndexToDimIndex(i);
    auto index_result = index_src;

    for (int d = 0; d < src.dim(); ++d)
    {
        // index_result[new_dims[d]] = index_src[d];
        index_result[d] = index_src[new_dims[d]];
    }
    result[index_result] = src[index_src];
}

void permute_impl(Tensor& src, Tensor& result, SizeType new_dims)
{
    CUDA_SWITCH_MACRO_ALL(src.scalar_type(), src.numel(), permute_impl, src, result, new_dims.vec());
}

template <typename TSource, typename TTarget>
__launch_bounds__(128) static __global__
    void copy_and_convert_impl(TensorInfoCuda<TSource> a, TensorInfoCuda<TTarget> b)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    if constexpr (std::is_same_v<TSource, int64_t>)
    {
        b[i] = TTarget((long long)a[i]);
    }
    else if constexpr (std::is_same_v<TTarget, int64_t>)
    {
        b[i] = TTarget((long long)a[i]);
    }
    else
    {
        b[i] = TTarget(a[i]);
    }
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
        case kFloat16:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), half, src.numel(), copy_and_convert_impl, src, target);
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
__launch_bounds__(128) static __global__ void rand_float_impl(TensorInfoCuda<T> a, float low, float high, uint64_t seed)
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
__launch_bounds__(128) static __global__ void rand_int_impl(TensorInfoCuda<T> a, int low, int high, uint64_t seed)
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
__launch_bounds__(128) static __global__ void sum_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
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
        CUDA_CASE_MACRO(sum_impl<half>, kFloat16, a.numel(), a, result)
        CUDA_CASE_MACRO(sum_impl<float>, kFloat, a.numel(), a, result)
        CUDA_CASE_MACRO(sum_impl<double>, kDouble, a.numel(), a, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}

template <typename T>
__launch_bounds__(128) static __global__ void sum_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
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
__launch_bounds__(128) static __global__
    void min_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MIN(a[i], b[i]);
}


template <typename T>
__launch_bounds__(128) static __global__ void set_min_result_impl(TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i == 0)
    {
        result[0] = +std::numeric_limits<T>::infinity();
    }
}

template <typename T>
__launch_bounds__(128) static __global__ void min_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;
    atomicMin(&result[i], a[i]);
}


void min_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), min_impl, a, b, result);
}
void min_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(result.scalar_type(), result.numel(), set_min_result_impl, result);
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), min_impl, a, result);
}


template <typename T>
__launch_bounds__(128) static __global__ void set_max_result_impl(TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i == 0)
    {
        result[0] = -std::numeric_limits<T>::infinity();
    }
}

template <typename T>
__launch_bounds__(128) static __global__ void max_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;
    atomicMax(&result[i], a[i]);
}


template <typename T>
__launch_bounds__(128) static __global__
    void max_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MAX(a[i], b[i]);
}

void max_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), max_impl, a, b, result);
}
void max_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(result.scalar_type(), result.numel(), set_max_result_impl, result);
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), max_impl, a, result);
}

template <typename T, typename TIndex>
__launch_bounds__(128) static __global__
    void index_select_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<TIndex> index, TensorInfoCuda<T> result)
{
    int64_t result_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (result_linear_index >= result.numel()) return;

    auto index_result    = result.LinearIndexToDimIndex(result_linear_index);
    auto index_input     = index_result;
    index_input[dim]     = index[index_result[dim]];
    result[index_result] = input[index_input];
}

template <typename TIndex>
static void index_select_helper(Tensor input, int64_t dim, TensorInfoCuda<TIndex> index, Tensor result)
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
    void index_add_impl(int64_t dim, TensorInfoCuda<TIndex> index, TensorInfoCuda<T> data, TensorInfoCuda<T> result)
{
    int64_t input_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (input_linear_index >= data.numel()) return;

    auto index_input  = data.LinearIndexToDimIndex(input_linear_index);
    auto index_result = index_input;
    CUDA_KERNEL_ASSERT(index_input[dim] < index.sizes[0]);
    index_result[dim] = index[index_input[dim]];
    CUDA_KERNEL_ASSERT(index_result[dim] < result.sizes[dim]);

    CUDA_KERNEL_ASSERT(result.index_in_range(index_result));
    CUDA_KERNEL_ASSERT(data.index_in_range(index_input));
    atomicAdd(&result[index_result], data[index_input]);
}

template <typename TIndex>
static void index_add_helper(Tensor data, int64_t dim, TensorInfoCuda<TIndex> index, Tensor result)
{
    CUDA_SWITCH_MACRO_FLOAT(result.scalar_type(), data.numel(), index_add_impl, dim, index, data, result);
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


template <typename T>
__launch_bounds__(128) static __global__
    void repeat_interleave_impl(TensorInfoCuda<T> input, int64_t count, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    auto index_result    = result.LinearIndexToDimIndex(i);
    auto index_input     = input.LinearIndexToDimIndex(i / count);
    result[index_result] = input[index_input];
}
void repeat_interleave_impl(Tensor input, int64_t count, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(result.scalar_type(), result.numel(), repeat_interleave_impl, input, count, result);
}


template <typename T>
__launch_bounds__(128) static __global__
    void transpose_impl(TensorInfoCuda<T> input, int64_t dim0, int64_t dim1, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    auto index_result = result.LinearIndexToDimIndex(i);
    auto index_input  = index_result;
    // swap(index_input[dim0], index_input[dim1]);
    auto t            = index_input[dim0];
    index_input[dim0] = index_input[dim1];
    index_input[dim1] = t;


    result[index_result] = input[index_input];
}
void transpose_impl(Tensor input, int64_t dim0, int64_t dim1, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(result.scalar_type(), result.numel(), transpose_impl, input, dim0, dim1, result);
}


}  // namespace cuda_impl
}  // namespace tinytorch