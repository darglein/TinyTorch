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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{

namespace cuda_impl
{

constexpr int REDUCE_BLOCK_SIZE = 256;


struct ReduceAdd
{
    template <typename T>
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a + b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicAddSelect(target, value);
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
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a * b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicMulSelect(target, value);
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
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a < b ? a : b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicMinSelect(target, value);
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
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a > b ? a : b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicMaxSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return std::numeric_limits<T>::lowest();
    }
};
struct StdHelper
{
    StdHelper(void* mean_ptr):mean_ptr(mean_ptr){}
    template <typename T>
    TT_HD T load_op(T a)
    {
        a = a - ((T*)mean_ptr)[0];
        return a * a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a + b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicAddSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return 0;
    }

    void* mean_ptr;

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

template <typename T, typename ShuffleType = int>
__device__ inline T shfl_up(T var, unsigned int srcLane, int width = 32)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_up_sync(0xFFFFFFFF, a[i], srcLane, width);
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
    __syncthreads();

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
    int64_t num_threads = std::min(a.numel(), int64_t(1024) * 1024);
    global_reduce<InputType, OutputType, Op>
        <<<iDivUp(num_threads, REDUCE_BLOCK_SIZE), REDUCE_BLOCK_SIZE, 0, cuda::getCurrentCUDAStream()>>>(
            a, result,op, Op::template default_value<OutputType>());
    CUDA_SYNC_CHECK_ERROR();
}


template <typename Op>
void global_reduce_helper(Tensor a, Tensor result, Op op)
{
    auto kernel_result = result;
    if (a.scalar_type() == kHalf)
    {
        kernel_result = result.to(kFloat);
    }

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
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }

    if (a.scalar_type() == kHalf)
    {
        result.copy_(kernel_result);
    }
}

void sum_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result,ReduceAdd());
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

// ====================================================================================================================



template <typename InputType, typename OutputType, typename Op>
static __global__ void dimensional_reduce(TensorInfoCuda<InputType> a, int64_t dim, TensorInfoCuda<OutputType> result,
                                          Op op, OutputType default_value)
{
    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;
    int64_t num_steps            = iDivUp(size_to_reduce, REDUCE_BLOCK_SIZE);

    // each reduction is computed by a separate block
    for (int64_t block_id = blockIdx.x; block_id < num_blocks_to_reduce; block_id += gridDim.x)
    {
        auto index_input = a.LinearIndexToDimIndexSkipDim(block_id, dim);
        OutputType value = default_value;


        for (int64_t k = 0; k < num_steps; ++k)
        {
            int64_t i = k * REDUCE_BLOCK_SIZE + threadIdx.x;
            index_input.set_index(dim, i);
            OutputType local_value = i < size_to_reduce ? OutputType(a[index_input]) : default_value;
            local_value            = blockReduce<REDUCE_BLOCK_SIZE, OutputType>(local_value, op, default_value);
            value                  = op(value, local_value);
        }
        if (threadIdx.x == 0)
        {
            CUDA_KERNEL_ASSERT(block_id < result.numel());
            result[block_id] = value;
        }
    }
}

template <typename InputType, typename OutputType, typename Op>
void dimensional_reduce_launcher(TensorInfoCuda<InputType> a, int64_t dim, TensorInfoCuda<OutputType> result)
{

    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;
    int64_t num_threads          = std::min(num_blocks_to_reduce * REDUCE_BLOCK_SIZE, int64_t(1024) * 1024);

    CHECK_EQ(result.numel(), num_blocks_to_reduce);

    dimensional_reduce<InputType, OutputType, Op>
        <<<iDivUp(num_threads, REDUCE_BLOCK_SIZE), REDUCE_BLOCK_SIZE, 0, cuda::getCurrentCUDAStream()>>>(
            a, dim, result, Op(), Op::template default_value<OutputType>());
    CUDA_SYNC_CHECK_ERROR();
}

template <typename Op>
void dimensional_reduce_helper(Tensor a, int64_t dim, Tensor result)
{
    // std::cout << "dimensional reduce " << dim << " | " << a.sizes() << " | " << result.sizes() << "\n";
    switch (a.scalar_type())
    {
        case kHalf:
            dimensional_reduce_launcher<__half, __half, Op>(a, dim, result);
            break;
        case kFloat:
            dimensional_reduce_launcher<float, float, Op>(a, dim, result);
            break;
        case kDouble:
            dimensional_reduce_launcher<double, double, Op>(a, dim, result);
            break;
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}

void sum_impl(Tensor a, int64_t dim, Tensor result)
{
    dimensional_reduce_helper<ReduceAdd>(a, dim, result);
}
void prod_impl(Tensor input, int64_t dim, Tensor result)
{
    dimensional_reduce_helper<ReduceProd>(input, dim, result);
}



// ========================================================================================================



template <typename T, typename Op, unsigned int LOCAL_WARP_SIZE = 32>
__device__ inline T warpInclusiveScan(T val, Op op, unsigned int lane)
{
#pragma unroll
    for (int d = 1; d < LOCAL_WARP_SIZE; d *= 2)
    {
        T tmp = shfl_up<T, T>(val, d, LOCAL_WARP_SIZE);
        if (lane >= d) val = op(val, tmp);
    }
    return val;
}


template <unsigned int BLOCK_SIZE, typename T, typename Op>
__device__ inline T blockInclusiveScan(T val, Op op, T default_value)
{
    __shared__ T shared[BLOCK_SIZE / 32];

    int lane_id   = threadIdx.x % 32;
    int warp_lane = threadIdx.x / 32;


    val = warpInclusiveScan(val, op, lane_id);

    // the last thread in the warp writes its value to shared memory
    if (lane_id == 31) shared[warp_lane] = val;
    __syncthreads();


    if (warp_lane == 0)
    {
        T valWarp;
        if (threadIdx.x < BLOCK_SIZE / 32)
        {
            valWarp = shared[threadIdx.x];
        }
        else
        {
            valWarp = default_value;
        }


        valWarp = warpInclusiveScan<T, Op>(valWarp, op, lane_id);
        if (threadIdx.x < BLOCK_SIZE / 32)
        {
            shared[lane_id] = valWarp;
        }
    }


    __syncthreads();

    // add value of previous warp
    if (warp_lane > 0)
    {
        val = op(val, shared[warp_lane - 1]);
    }

    __syncthreads();

    return val;
}



template <typename InputType, typename OutputType, typename Op>
static __global__ void dimensional_scan(TensorInfoCuda<InputType> a, int64_t dim, TensorInfoCuda<OutputType> result,
                                        Op op, OutputType default_value)
{
    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;
    int64_t num_steps            = iDivUp(size_to_reduce, REDUCE_BLOCK_SIZE);
    __shared__ OutputType previous_block_sum;



    // each reduction is computed by a separate block
    for (int64_t block_id = blockIdx.x; block_id < num_blocks_to_reduce; block_id += gridDim.x)
    {
        auto index_input = a.LinearIndexToDimIndexSkipDim(block_id, dim);

        if (threadIdx.x == 0)
        {
            previous_block_sum = default_value;
        }
        __syncthreads();

        for (int64_t k = 0; k < num_steps; ++k)
        {
            auto previous_block_sum_copy = previous_block_sum;
            int64_t i                    = k * REDUCE_BLOCK_SIZE + threadIdx.x;
            index_input.set_index(dim, i);
            OutputType local_value = i < size_to_reduce ? OutputType(a[index_input]) : default_value;
            auto value = blockInclusiveScan<REDUCE_BLOCK_SIZE, OutputType, Op>(local_value, op, default_value);
            value      = op(value, previous_block_sum_copy);

            if (i < size_to_reduce)
            {
                result[index_input] = value;
            }

            if (threadIdx.x == REDUCE_BLOCK_SIZE - 1)
            {
                previous_block_sum = value;
            }
            __syncthreads();
        }
    }
}

template <typename InputType, typename OutputType, typename Op>
void dimensional_scan_launcher(TensorInfoCuda<InputType> a, int64_t dim, TensorInfoCuda<OutputType> result)
{
    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;
    int64_t num_threads          = std::min(num_blocks_to_reduce * REDUCE_BLOCK_SIZE, int64_t(1024) * 1024);



    dimensional_scan<InputType, OutputType, Op>
        <<<iDivUp(num_threads, REDUCE_BLOCK_SIZE), REDUCE_BLOCK_SIZE, 0, cuda::getCurrentCUDAStream()>>>(
            a, dim, result, Op(), Op::template default_value<OutputType>());
    CUDA_SYNC_CHECK_ERROR();
}

template <typename Op>
void dimensional_scan_helper(Tensor a, int64_t dim, Tensor result)
{
    switch (a.scalar_type())
    {
        case kHalf:
            dimensional_scan_launcher<__half, __half, Op>(a, dim, result);
            break;
        case kInt32:
            dimensional_scan_launcher<int, int, Op>(a, dim, result);
            break;
        case kFloat:
            dimensional_scan_launcher<float, float, Op>(a, dim, result);
            break;
        case kDouble:
            dimensional_scan_launcher<double, double, Op>(a, dim, result);
            break;
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}
void cumsum_impl(Tensor input, int64_t dim, Tensor result)
{
    dimensional_scan_helper<ReduceAdd>(input, dim, result);
}

void cumprod_impl(Tensor input, int64_t dim, Tensor result)
{
    dimensional_scan_helper<ReduceProd>(input, dim, result);
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
            atomicMinSelect(&result_value, v);
        }
        else
        {
            atomicMaxSelect(&result_value, v);
        }
    }
}
void min_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    CUDA_SWITCH_MACRO_FLOAT(input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, true);
    indices = Tensor();
}

void max_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    CUDA_SWITCH_MACRO_FLOAT(input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, false);
    indices = Tensor();
}
}  // namespace cuda_impl
}  // namespace tinytorch