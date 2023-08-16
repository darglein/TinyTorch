/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops/ops_impl.h"
#include "torch/cuda/atomic_minmax.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include "unary_operators.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{

namespace cuda_impl
{

constexpr int REDUCE_BLOCK_SIZE = 256;

template <int Size>
struct AtomicCASType
{
};

template <>
struct AtomicCASType<2>
{
    using Type = short;
};
template <>
struct AtomicCASType<4>
{
    using Type = int;
};
template <>
struct AtomicCASType<8>
{
    using Type = unsigned long long int;
};


struct ReduceAdd
{
    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a + b;
    }
    template <typename T>
    TT_HD void atomic_reduce(T* target, T value)
    {
        atomicAdd(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return T(0);
    }
};

struct ReduceProd
{
    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a * b;
    }
    template <typename T>
    static constexpr T default_value()
    {
        return T(1);
    }
};
struct ReduceMin
{
    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a < b ? a : b;
    }
    template <typename T>
    TT_HD void atomic_reduce(T* target, T value)
    {
        atomicMin(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return std::numeric_limits<T>::max();
    }
};
struct ReduceMax
{
    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a > b ? a : b;
    }
    template <typename T>
    TT_HD void atomic_reduce(T* target, T value)
    {
        atomicMax(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return std::numeric_limits<T>::lowest();
    }
};

template <typename T, typename ShuffleType = int>
__device__ inline T shfl_xor(T var, unsigned int srcLane, int width = 32)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_xor_sync(0xFFFFFFFF, a[i], srcLane, width);
    }
    return var;
}

template <typename T, typename OP, unsigned int LOCAL_WARP_SIZE = 32>
__device__ inline T warpReduce(T val, OP op)
{
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = shfl_xor<T, T>(val, offset);
        val    = op(val, v);
    }
    return val;
}


template <int BLOCK_SIZE, typename T, typename OP>
__device__ inline T blockReduce(T val, OP op, T default_val)
{
    __shared__ T shared[BLOCK_SIZE / 32];

    int lane   = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;

    // Each warp reduces with registers
    val = warpReduce(val, op);

    // The first thread in each warp writes to smem
    if (lane == 0)
    {
        shared[warpid] = val;
    }

    __syncthreads();

    if (warpid == 0)
    {
        if (threadIdx.x < BLOCK_SIZE / 32)
        {
            val = shared[threadIdx.x];
        }
        else
        {
            val = default_val;
        }


        val = warpReduce(val, op);
    }


    return val;
}

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
        OutputType local_value = i < a.numel() ? OutputType(a[i]) : default_value;
        local_value            = blockReduce<REDUCE_BLOCK_SIZE, OutputType>(local_value, op, default_value);
        value                  = op(value, local_value);
    }
    if (threadIdx.x == 0)
    {
        op.atomic_reduce(&result[0], value);
    }
}

template <typename Op>
void global_reduce_helper(Tensor a, Tensor& result)
{
    int64_t num_threads = std::min(a.numel(), int64_t(1024) * 1024);
    switch (a.scalar_type())
    {
        CUDA_CASE_MACRO_REFINED(REDUCE_BLOCK_SIZE, (global_reduce<int, int>), kInt32, num_threads, a, result, Op(),
                                Op::template default_value<int>())
        CUDA_CASE_MACRO_REFINED(REDUCE_BLOCK_SIZE, (global_reduce<__half, __half>), kFloat16, num_threads, a, result,
                                Op(), Op::template default_value<__half>())
        CUDA_CASE_MACRO_REFINED(REDUCE_BLOCK_SIZE, (global_reduce<float, float>), kFloat, num_threads, a, result, Op(),
                                Op::template default_value<float>())
        CUDA_CASE_MACRO_REFINED(REDUCE_BLOCK_SIZE, (global_reduce<double, double>), kDouble, num_threads, a, result,
                                Op(), Op::template default_value<double>())
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}

void sum_impl(Tensor a, Tensor& result)
{
    global_reduce_helper<ReduceAdd>(a, result);
}
void min_impl(Tensor a, Tensor& result)
{
    global_reduce_helper<ReduceMin>(a, result);
}
void max_impl(Tensor a, Tensor& result)
{
    global_reduce_helper<ReduceMax>(a, result);
}


// ====================================================================================================================



template <typename InputType, typename OutputType, typename Op>
static __global__ void dimensional_reduce(TensorInfoCuda<InputType> a, int64_t dim, TensorInfoCuda<OutputType> result,
                                          Op op, OutputType default_value)
{
    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;
    int64_t num_steps = iDivUp(size_to_reduce,REDUCE_BLOCK_SIZE);

    // each reduction is computed by a separate block
    for (int64_t block_id = blockIdx.x; block_id < num_blocks_to_reduce; block_id += gridDim.x)
    {
        auto index_input = a.LinearIndexToDimIndexSkipDim(block_id, dim);
        OutputType value = default_value;


        for (int64_t k = 0; k < num_steps; ++k)
        {
            int64_t i              = k * REDUCE_BLOCK_SIZE + threadIdx.x;
            index_input.set_index(dim,i);
            OutputType local_value = i < size_to_reduce ? OutputType(a[index_input]) : default_value;
            local_value            = blockReduce<REDUCE_BLOCK_SIZE, OutputType>(local_value, op, default_value);
            value                  = op(value, local_value);
        }
        if (threadIdx.x == 0)
        {
           result[block_id] = value;
        }
    }
}

template <typename Op>
void dimensional_reduce_helper(Tensor a, int64_t dim, Tensor& result)
{
    int64_t num_threads = std::min(a.numel(), int64_t(1024) * 1024);
    switch (a.scalar_type())
    {
        CUDA_CASE_MACRO_REFINED(REDUCE_BLOCK_SIZE, (dimensional_reduce<__half, __half>), kHalf, num_threads, a, dim,
                                result, Op(), Op::template default_value<__half>())
        CUDA_CASE_MACRO_REFINED(REDUCE_BLOCK_SIZE, (dimensional_reduce<float, float>), kFloat, num_threads, a, dim,
                                result, Op(), Op::template default_value<float>())
        CUDA_CASE_MACRO_REFINED(REDUCE_BLOCK_SIZE, (dimensional_reduce<double, double>), kDouble, num_threads, a, dim,
                                result, Op(), Op::template default_value<double>())
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
//    cudaDeviceSynchronize();
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
    dimensional_reduce_helper<ReduceAdd>(a,dim,result);
    return;
    auto stream = cuda::getCurrentCUDAStream();
    switch (a.scalar_type())
    {
        CUDA_CASE_MACRO(sum_impl<int32_t>, kInt32, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<half>, kHalf, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<float>, kFloat, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<double>, kDouble, a.numel(), a, dim, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}



template <typename T>
__launch_bounds__(128) static __global__ void prod_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    {
        auto index_result = result.LinearIndexToDimIndex(i);

        T prod = T(1.f);
        for (int64_t j = 0; j < input.size(dim); ++j)
        {
            auto index_input = index_result;
            index_input[dim] = j;
            prod             = prod * input[index_input];
        }
        result[index_result] = prod;
    }
}

void prod_impl(Tensor input, int64_t dim, Tensor& result)
{
    dimensional_reduce_helper<ReduceProd>(input,dim,result);
    return;
//    CUDA_SWITCH_MACRO_ALL(input.scalar_type(), result.numel(), prod_impl, input, dim, result);
}
template <typename T>
__launch_bounds__(128) static __global__
    void cumprod_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    {
        auto index_result = result.LinearIndexToDimIndex(i);

        T prod = T(1.f);
        for (int64_t j = 0; j <= index_result[dim]; ++j)
        {
            auto index_input = index_result;
            index_input[dim] = j;
            prod             = prod * input[index_input];
        }
        result[index_result] = prod;
    }
}
void cumprod_impl(Tensor input, int64_t dim, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(input.scalar_type(), result.numel(), cumprod_impl, input, dim, result);
}
template <typename T>
__launch_bounds__(128) static __global__
    void cumsum_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    {
        auto index_result = result.LinearIndexToDimIndex(i);

        T prod = T(0.f);
        for (int64_t j = 0; j <= index_result[dim]; ++j)
        {
            auto index_input = index_result;
            index_input[dim] = j;
            prod             = prod + input[index_input];
        }
        result[index_result] = prod;
    }
}
void cumsum_impl(Tensor input, int64_t dim, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(input.scalar_type(), result.numel(), cumsum_impl, input, dim, result);
}



template <typename T>
__launch_bounds__(128) static __global__
    void min_max_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<int64_t> indices, TensorInfoCuda<T> result,
                      bool calc_min)
{
    using G = typename CpuComputeFloatType<T>::Type;


    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= input.numel()) return;

    {
        G v               = input[i];
        auto index_input  = input.LinearIndexToDimIndex(i);
        auto index_result = index_input;
        index_result[dim] = 0;

        auto& result_value = result[index_result];
        auto& result_index = indices[index_result];


        if (calc_min)
        {
            atomicMin(&result_value, v);
        }
        else
        {
            atomicMax(&result_value, v);
        }
    }
}
void min_impl(Tensor input, int64_t dim, Tensor& result, Tensor& indices)
{
    CUDA_SWITCH_MACRO_FLOAT(input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, true);
    indices = Tensor();
}

void max_impl(Tensor input, int64_t dim, Tensor& result, Tensor& indices)
{
    CUDA_SWITCH_MACRO_FLOAT(input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, false);
    indices = Tensor();
}
}  // namespace cuda_impl
}  // namespace tinytorch