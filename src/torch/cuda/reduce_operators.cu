/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "reduce_helper.h"
#include "torch/core/ops/ops_impl.h"
#include "torch/cuda/atomic_minmax.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include "unary_operators.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tinytorch
{

namespace cuda_impl
{


template <typename InputType, typename OutputType, typename Op>
static __global__ void global_reduce(TensorInfoCuda<InputType> a, TensorInfoCuda<OutputType> result, Op op,
                                     OutputType default_value)
{
    int64_t grid_size = blockDim.x * gridDim.x;
    int64_t num_steps = iDivUp(a.numel(), grid_size);

    OutputType value = default_value;
    for (int64_t k = 0; k < num_steps; ++k)
    {
        int64_t i              = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x + k * grid_size;
        OutputType local_value = i < a.numel() ? op.load_op(OutputType(a[i])) : default_value;
        local_value            = blockReduce<REDUCE_BLOCK_SIZE, OutputType>(local_value, op, default_value);
        value                  = op(value, local_value);
    }
    if (threadIdx.x == 0)
    {
        op.atomic_reduce(&result[0], value);
    }
}
template <typename InputType, typename OutputType, typename Op>
void global_reduce_launcher(TensorInfoCuda<InputType> a, TensorInfoCuda<OutputType> result, Op op)
{
    int64_t num_threads = std::min(int64_t(a.numel()), int64_t(1024) * 1024);
    global_reduce<InputType, OutputType, Op>
        <<<iDivUp(num_threads, REDUCE_BLOCK_SIZE), REDUCE_BLOCK_SIZE, 0, cuda::getCurrentCUDAStream()>>>(
            a, result, op, Op::template default_value<OutputType>());
    CUDA_SYNC_CHECK_ERROR();
}


template <typename Op>
void global_reduce_helper(Tensor a, Tensor result, Op op)
{
    auto kernel_result = result;
    if (a.scalar_type() == kHalf || a.scalar_type() == kUInt16)
    {
        kernel_result = result.to(kFloat);
    }

    cuda::DeviceGuard guard(a.device());
    switch (a.scalar_type())
    {
        case kInt32:
            global_reduce_launcher<int, int, Op>(a, kernel_result, op);
            break;
        case kInt64:
            global_reduce_launcher<int64_t, int64_t, Op>(a, kernel_result, op);
            break;
        case kFloat16:
            global_reduce_launcher<__half, float, Op>(a, kernel_result, op);
            break;
        case kFloat:
            global_reduce_launcher<float, float, Op>(a, kernel_result, op);
            break;
        case kDouble:
            global_reduce_launcher<double, double, Op>(a, kernel_result, op);
            break;
        case kUInt16:
            global_reduce_launcher<uint16_t, float, Op>(a, kernel_result, op);
            break;
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }

    if (a.scalar_type() == kHalf || a.scalar_type() == kUInt16)
    {
        result.copy_(kernel_result);
    }
}
void abs_sum_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceAbsAdd());
}
void prod_sum_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceProdAdd());
}
void sum_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceAdd());
}
void min_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceMin());
}
void max_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceMax());
}

void std_helper_impl(Tensor a, Tensor mean, Tensor result)
{
    global_reduce_helper(a, result, StdHelper(mean.data_ptr()));
}


// Monotone key: smaller floating point value <-> smaller key (total order, NaNs sort last)
template <typename T>
__device__ inline unsigned long long min_key(T v)
{
    if (std::is_same<T, double>::value)
    {
        unsigned long long b = __double_as_longlong(v);
        return (b & 0x8000000000000000ULL) ? ~b : (b | 0x8000000000000000ULL);
    }
    if (std::is_same<T, __half>::value)
    {
        v = __half2float(v);
    }
    float f      = float(v);
    unsigned int b = __float_as_uint(f);
    unsigned int k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return (unsigned long long)k;
}

template <typename T>
__device__ inline T unkey(unsigned long long k)
{
    if (std::is_same<T, double>::value)
    {
        unsigned long long b = (k & 0x8000000000000000ULL) ? (k ^ 0x8000000000000000ULL) : ~k;
        return T(__longlong_as_double(b));
    }
    unsigned int kk = (unsigned int)k;
    unsigned int b  = (kk & 0x80000000u) ? (kk ^ 0x80000000u) : ~kk;
    float f         = __uint_as_float(b);
    if (std::is_same<T, __half>::value)
    {
        return __float2half(f);
    }
    return T(f);
}

// scatter: one thread per input element, atomically selects (value, index) per output slot
template <typename T>
__launch_bounds__(128) static __global__
    void min_max_impl(TensorInfoCuda<T> input, TensorInfoCuda<T> result, int64_t dim, unsigned long long* keys,
                      bool calc_min)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= input.numel()) return;

    T v               = input[i];
    auto index_input  = input.LinearIndexToDimIndex(i);
    int64_t idx       = index_input[dim];
    auto index_result = index_input;
    index_result[dim] = 0;

    unsigned long long packed = (min_key<T>(v) << 32) | (unsigned int)idx;
    if (calc_min)
    {
        atomicMin(keys + result.IndexToOffset(index_result), packed);
    }
    else
    {
        atomicMax(keys + result.IndexToOffset(index_result), packed);
    }
}

// gather: one thread per output slot, unpacks (key, index) into result/indices
template <typename T>
__launch_bounds__(128) static __global__
    void min_max_unpack_impl(TensorInfoCuda<T> result, TensorInfoCuda<int64_t> indices, const unsigned long long* keys)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    unsigned long long packed = keys[i];
    result[i]                 = unkey<T>(packed >> 32);
    indices[i]                = (int64_t)(packed & 0xFFFFFFFFULL);
}

template <typename TT>
static void min_max_run(Tensor input, int64_t dim, Tensor result, Tensor indices, unsigned long long* keys_ptr,
                        int64_t numel, int64_t out_numel, bool calc_min)
{
    min_max_impl<TT><<<iDivUp(numel, 128), 128, 0, cuda::getCurrentCUDAStream()>>>(
        TensorInfoCuda<TT>(input), TensorInfoCuda<TT>(result), dim, keys_ptr, calc_min);
    min_max_unpack_impl<TT><<<iDivUp(out_numel, 128), 128, 0, cuda::getCurrentCUDAStream()>>>(
        TensorInfoCuda<TT>(result), TensorInfoCuda<int64_t>(indices), keys_ptr);
}

static void min_max_dispatch(Tensor input, int64_t dim, Tensor result, Tensor indices, bool calc_min)
{
    int64_t out_numel = result.numel();
    int64_t numel     = input.numel();

    // scratch buffer holding packed (monotone key << 32 | index)
    Tensor keys = empty(result.sizes(), TensorOptions().dtype(kLong).device(input.device()));
    unsigned long long* keys_ptr = reinterpret_cast<unsigned long long*>(keys.data_ptr<int64_t>());
    TT_CHECK_CUDA_ERROR(cudaMemsetAsync(keys_ptr, calc_min ? 0xFF : 0x00, sizeof(unsigned long long) * (size_t)out_numel,
                                        cuda::getCurrentCUDAStream()));

    switch (input.scalar_type())
    {
        case kHalf:
            min_max_run<__half>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kFloat:
            min_max_run<float>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kDouble:
            min_max_run<double>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        default:
            CHECK(false) << "invalid input type " << (int)input.scalar_type();
    }
}

void min_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    cuda::DeviceGuard guard(input.device());
    if (input.numel() > 0 && result.numel() > 0)
    {
        min_max_dispatch(input, dim, result, indices, true);
    }
}

void max_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    cuda::DeviceGuard guard(input.device());
    if (input.numel() > 0 && result.numel() > 0)
    {
        min_max_dispatch(input, dim, result, indices, false);
    }
}
}  // namespace cuda_impl
}  // namespace tinytorch