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

template <typename T>
__launch_bounds__(128)
static __global__ void range_impl_cuda(TensorInfo<T> a, double start, double end, double step)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(start) + T(i) * T(step);
}

void range_impl_cuda(Tensor a, double start, double end, double step)
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), range_impl_cuda, a, start, end, step);
}

template <typename T>
__launch_bounds__(128)
static __global__ void fill_impl_cuda(TensorInfo<T> a, double value)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(value);
}

void fill_impl_cuda(Tensor a, double value)
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), fill_impl_cuda, a, value);
}

template <typename TSource, typename TTarget>
__launch_bounds__(128)
static __global__ void copy_and_convert_impl_cuda(TensorInfo<TSource> a, TensorInfo<TTarget> b)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    b[i] = TTarget(a[i]);
}

template <typename TSource>
static void copy_and_convert_helper_cuda(TensorInfo<TSource> src, Tensor target)
{
    auto stream = cuda::getCurrentCUDAStream();
    int64_t numel = src.numel();
    
    switch (target.scalar_type())
    {
        case kUInt8:
            copy_and_convert_impl_cuda<TSource, uint8_t><<<iDivUp(numel, 128), 128, 0, stream>>>(src, target);
            break;
        case kInt16:
            copy_and_convert_impl_cuda<TSource, int16_t><<<iDivUp(numel, 128), 128, 0, stream>>>(src, target);
            break;
        case kInt32:
            copy_and_convert_impl_cuda<TSource, int32_t><<<iDivUp(numel, 128), 128, 0, stream>>>(src, target);
            break;
        case kLong:
            copy_and_convert_impl_cuda<TSource, int64_t><<<iDivUp(numel, 128), 128, 0, stream>>>(src, target);
            break;
        case kFloat:
            copy_and_convert_impl_cuda<TSource, float><<<iDivUp(numel, 128), 128, 0, stream>>>(src, target);
            break;
        case kDouble:
            copy_and_convert_impl_cuda<TSource, double><<<iDivUp(numel, 128), 128, 0, stream>>>(src, target);
            break;
        default:
            CHECK(false) << "invalid input type " << target.scalar_type();
    }
}

void copy_and_convert_impl_cuda(Tensor src, Tensor& target)
{
    switch (src.scalar_type())
    {
        case kUInt8:
            copy_and_convert_helper_cuda<uint8_t>(src, target);
            break;
        case kInt16:
            copy_and_convert_helper_cuda<int16_t>(src, target);
            break;
        case kInt32:
            copy_and_convert_helper_cuda<int32_t>(src, target);
            break;
        case kLong:
            copy_and_convert_helper_cuda<int64_t>(src, target);
            break;
        case kFloat:
            copy_and_convert_helper_cuda<float>(src, target);
            break;
        case kDouble:
            copy_and_convert_helper_cuda<double>(src, target);
            break;
        default:
            CHECK(false) << "invalid input type " << src.scalar_type();
    }
}


template <typename T>
__launch_bounds__(128) 
static __global__ void rand_float_impl_cuda(TensorInfo<T> a, float low, float high, uint64_t seed)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    curandState state;
    curand_init(seed, i, 0, &state);
    a[i] = T(curand_uniform(&state) * (high - low) + low);
}

void uniform_impl_cuda(Tensor& a, double mi, double ma)
{
    static int64_t seed = get_seed();
    srand((uint32_t)seed);
    seed = ::rand();
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), rand_float_impl_cuda, a, (float)mi, (float)ma, seed);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void rand_int_impl_cuda(TensorInfo<T> a, int low, int high, uint64_t seed)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    curandState state;
    curand_init(seed, i, 0, &state);
    a[i] = T(curand(&state) % (high - low) + low);
}

void uniform_int_impl_cuda(Tensor& a, int low, int high)
{
    static int64_t seed = get_seed();
    srand((uint32_t)seed);
    seed = ::rand();
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), rand_int_impl_cuda, a, low, high, seed);
}

template <typename T>
__launch_bounds__(128)
static __global__ void sum_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    T* ptr = &result[0];
    atomicAdd(ptr, a[i]);
}

void sum_impl_cuda(Tensor a, Tensor& result)
{
    auto stream = cuda::getCurrentCUDAStream();
    switch (a.scalar_type())
    {
        CASE_MACRO(sum_impl_cuda, int32_t, kInt32, a.numel(), a, result)
        CASE_MACRO(sum_impl_cuda, float, kFloat, a.numel(), a, result)
        CASE_MACRO(sum_impl_cuda, double, kDouble, a.numel(), a, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}

void sum_impl_cuda(Tensor a, int64_t dim, Tensor& result) 
{
    throw std::runtime_error("not implemented");
}

template <typename T>
__launch_bounds__(128)
static __global__ void log_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::log(a[i]);
}

void log_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), log_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void log1p_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::log1p(a[i]);
}

void log1p_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), log1p_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void exp_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::exp(a[i]);
}

void exp_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), exp_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void sign_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    T v       = a[i];
    result[i] = (v < T(0)) ? T(-1) : (v > T(0)) ? T(1) : T(0);
}

void sign_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sign_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void pow_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = T(std::pow(a[i], b));
}

void pow_impl_cuda(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), pow_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void sin_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::sin(a[i]);
}

void sin_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sin_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void cos_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::cos(a[i]);
}

void cos_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), cos_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void relu_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = relu(a[i]);
}

void relu_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), relu_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void sigmoid_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = sigmoid(a[i]);
}

void sigmoid_impl_cuda(Tensor a, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sigmoid_impl_cuda, a, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void softplus_impl_cuda(TensorInfo<T> a, double beta, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = softplus(a[i], T(beta));
}

void softplus_impl_cuda(Tensor a, double beta, Tensor& result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), softplus_impl_cuda, a, beta, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void min_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MIN(a[i], b[i]);
}

void min_impl_cuda(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), min_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128)
static __global__ void max_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MAX(a[i], b[i]);
}

void max_impl_cuda(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), max_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void index_select_impl_cuda(TensorInfo<T> input, int64_t dim, TensorInfo<int32_t> index, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims    = input.dims;
    int64_t slice_size = input.numel() / input.sizes[dim];

    int64_t index_index = i / slice_size;
    int64_t slice        = index[index_index];
    int64_t input_start  = slice * input.strides[dim];
    int64_t result_start = index_index * result.strides[dim];

    int64_t c = i % slice_size;


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

    // result[i] should be the same as result.data[result_offset]
    result.data[result_offset] = input.data[input_offset];
}

void index_select_impl_cuda(Tensor input, int64_t dim, Tensor index, Tensor& result) 
{
    SWITCH_MACRO_ALL(result.scalar_type(), result.numel(), index_select_impl_cuda, input, dim, index, result);
}






}  // namespace tinytorch
