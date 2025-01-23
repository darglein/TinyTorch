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



// ====================================================================================================================



template <typename InputType, typename OutputType, typename Op, int REDUCE_BLOCK_SIZE, int MAX_DIMS, int REDUCE_DIM>
static __global__ void dimensional_reduce_single_thread(TensorInfoCuda<InputType, MAX_DIMS> a, int64_t dynamic_dim,
                                                        TensorInfoCuda<OutputType, MAX_DIMS> result, Op op,
                                                        OutputType default_value)
{
    int64_t dim = REDUCE_DIM == -1 ? dynamic_dim : REDUCE_DIM;

    using IndexType                = typename TensorInfoCuda<InputType, MAX_DIMS>::IndexType;
    IndexType size_to_reduce       = a.size(dim);
    IndexType num_blocks_to_reduce = a.numel() / size_to_reduce;

    int64_t tid = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;

    if (tid < num_blocks_to_reduce)
    {
        auto index_input = a.LinearIndexToDimIndexSkipDim(tid, dim);
        OutputType value = default_value;
        for (IndexType k = 0; k < size_to_reduce; ++k)
        {
            index_input.set_index(dim, k);
            OutputType local_value = op.load_op(OutputType(a[index_input]));
            value                  = op(value, local_value);
        }
        result[tid] = value;
    }
}

template <typename InputType, typename OutputType, typename Op, int REDUCE_BLOCK_SIZE, int MAX_DIMS, int REDUCE_DIM>
static __global__ void dimensional_reduce(TensorInfoCuda<InputType, MAX_DIMS> a, int64_t dynamic_dim,
                                          TensorInfoCuda<OutputType, MAX_DIMS> result, Op op, OutputType default_value)
{
    int64_t dim = REDUCE_DIM == -1 ? dynamic_dim : REDUCE_DIM;

    using IndexType                = typename TensorInfoCuda<InputType, MAX_DIMS>::IndexType;
    IndexType size_to_reduce       = a.size(dim);
    IndexType num_blocks_to_reduce = a.numel() / size_to_reduce;
    IndexType num_steps            = iDivUp(size_to_reduce, REDUCE_BLOCK_SIZE);

    // each reduction is computed by a separate block
    for (IndexType block_id = blockIdx.x; block_id < num_blocks_to_reduce; block_id += gridDim.x)
    {
        auto index_input = a.LinearIndexToDimIndexSkipDim(block_id, dim);
        OutputType value = default_value;


        for (IndexType k = 0; k < num_steps; ++k)
        {
            IndexType i = k * REDUCE_BLOCK_SIZE + threadIdx.x;
            index_input.set_index(dim, i);
            OutputType local_value = i < size_to_reduce ? op.load_op(OutputType(a[index_input])) : default_value;

            if constexpr (REDUCE_BLOCK_SIZE <= 32)
            {
                local_value = warpReduce<OutputType, Op, REDUCE_BLOCK_SIZE>(local_value, op);
            }
            else
            {
                local_value = blockReduce<REDUCE_BLOCK_SIZE, OutputType>(local_value, op, default_value);
            }
            value = op(value, local_value);
        }
        if (threadIdx.x == 0)
        {
            CUDA_KERNEL_ASSERT(block_id < result.numel());
            result[block_id] = value;
        }
    }
}


template <typename InputType, typename OutputType, typename Op, int MAX_DIMS, int REDUCE_DIM>
void dimensional_reduce_launcher_size_dim(TensorInfoCuda<InputType, MAX_DIMS> a, int64_t dim,
                                          TensorInfoCuda<OutputType, MAX_DIMS> result)
{
    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;

    CHECK_EQ(result.numel(), num_blocks_to_reduce);

    if (a.contiguous && size_to_reduce < 16 && num_blocks_to_reduce > 20000 && dim == 0)
    {
        // this is the "simple" case, where just running one thread per reduce block is the most efficient
        int64_t num_threads = num_blocks_to_reduce;

        dimensional_reduce_single_thread<InputType, OutputType, Op, REDUCE_BLOCK_SIZE, MAX_DIMS, REDUCE_DIM>
            <<<iDivUp(num_threads, REDUCE_BLOCK_SIZE), REDUCE_BLOCK_SIZE, 0, cuda::getCurrentCUDAStream()>>>(
                a, dim, result, Op(), Op::template default_value<OutputType>());
    }
    else if (size_to_reduce > 8)
    {
        int64_t num_threads = std::min(num_blocks_to_reduce * REDUCE_BLOCK_SIZE, int64_t(1024) * 1024);
        dimensional_reduce<InputType, OutputType, Op, REDUCE_BLOCK_SIZE, MAX_DIMS, REDUCE_DIM>
            <<<iDivUp(num_threads, REDUCE_BLOCK_SIZE), REDUCE_BLOCK_SIZE, 0, cuda::getCurrentCUDAStream()>>>(
                a, dim, result, Op(), Op::template default_value<OutputType>());
    }
    else
    {
        int64_t num_threads = std::min(num_blocks_to_reduce * REDUCE_BLOCK_SIZE, int64_t(1024) * 1024);
        dimensional_reduce<InputType, OutputType, Op, 8, MAX_DIMS, REDUCE_DIM>
            <<<iDivUp(num_threads, 8), 8, 0, cuda::getCurrentCUDAStream()>>>(a, dim, result, Op(),
                                                                             Op::template default_value<OutputType>());
    }
    CUDA_SYNC_CHECK_ERROR();
}

template <typename InputType, typename OutputType, typename Op, int MAX_DIMS>
void dimensional_reduce_launcher_size(Tensor a, int64_t dim, Tensor result)
{
    cuda::DeviceGuard guard(a.device());
    switch (dim)
    {
        case 0:
            dimensional_reduce_launcher_size_dim<InputType, OutputType, Op, MAX_DIMS, 0>(a, dim, result);
            break;
        case 1:
            dimensional_reduce_launcher_size_dim<InputType, OutputType, Op, MAX_DIMS, 1>(a, dim, result);
            break;
        default:
            dimensional_reduce_launcher_size_dim<InputType, OutputType, Op, MAX_DIMS, -1>(a, dim, result);
            break;
    }
}

template <typename InputType, typename OutputType, typename Op>
void dimensional_reduce_launcher(Tensor a, int64_t dim, Tensor result)
{
    if (a.dim() == 2 && result.dim() == 2)
    {
        dimensional_reduce_launcher_size<InputType, OutputType, Op, 2>(a, dim, result);
    }
    else
    {
        dimensional_reduce_launcher_size<InputType, OutputType, Op, -1>(a, dim, result);
    }
}


template <typename Op>
void dimensional_reduce_helper(Tensor a, int64_t dim, Tensor result)
{
    cuda::DeviceGuard guard(a.device());
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

void abs_sum_impl(Tensor a, int64_t dim, Tensor result)
{
    dimensional_reduce_helper<ReduceAbsAdd>(a, dim, result);
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
    cuda::DeviceGuard guard(a.device());
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

}  // namespace cuda_impl
}  // namespace tinytorch