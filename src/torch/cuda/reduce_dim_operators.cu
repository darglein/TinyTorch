/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <algorithm>
#include <limits>

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


// A dimensional reduction computes num_outputs = numel / size(dim) results. Few long reductions (e.g. one sum per
// image) are split into num_splits parts, which the first pass reduces into partials[output * num_splits + split] and
// the second pass (dimensional_reduce_partials) combines in a fixed order, so the result stays deterministic.
// Otherwise a large image summed in a single block used one SM of the GPU.

// The minimum number of elements of one split: a block-wide split reads at least 16 elements per thread, a split of
// a thread-per-output reduction at least 64.
constexpr int64_t REDUCE_MIN_SPLIT_BLOCK  = 16 * REDUCE_BLOCK_SIZE;
constexpr int64_t REDUCE_MIN_SPLIT_THREAD = 64;
constexpr int64_t REDUCE_MAX_SPLITS       = 1024;

// The number of (output, split) units that should run in parallel: enough blocks/threads to fill a large GPU. They are
// fixed instead of derived from the SM count, so a float reduction gives the same bits on every GPU.
constexpr int64_t REDUCE_TARGET_BLOCKS  = 1024;
constexpr int64_t REDUCE_TARGET_THREADS = 256 * 1024;

// ceil(a / b) in 64 bit: iDivUp returns uint32_t, which truncates the split length of huge views and the grid size
// that the launcher checks
TT_HD constexpr int64_t reduce_div_up(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

// Writes the reduced value of one (output, split) pair: the result if the reduction is not split, else its partial.
template <typename InputType, typename AccType, int MAX_DIMS>
__device__ inline void dimensional_reduce_store(TensorInfoCuda<InputType, MAX_DIMS>& result, AccType* partials,
                                                int64_t num_splits, int64_t output, int64_t split, AccType value)
{
    if (num_splits == 1)
    {
        CUDA_KERNEL_ASSERT(output < result.numel());
        result[output] = InputType(value);
    }
    else
    {
        partials[output * num_splits + split] = value;
    }
}

// One thread per (output, split). Used for short reductions and if the reduced dimension is not the innermost one:
// adjacent threads then compute adjacent outputs, which are adjacent in memory.
template <typename InputType, typename AccType, typename Op, int MAX_DIMS, int REDUCE_DIM>
static __global__ void dimensional_reduce_thread(TensorInfoCuda<InputType, MAX_DIMS> a, int64_t dynamic_dim,
                                                 TensorInfoCuda<InputType, MAX_DIMS> result, AccType* partials,
                                                 int64_t num_splits, Op op, AccType default_value)
{
    int64_t dim = REDUCE_DIM == -1 ? dynamic_dim : REDUCE_DIM;

    using IndexType          = typename TensorInfoCuda<InputType, MAX_DIMS>::IndexType;
    IndexType size_to_reduce = a.size(dim);
    IndexType num_outputs    = a.numel() / size_to_reduce;
    IndexType split_length   = reduce_div_up(size_to_reduce, num_splits);

    IndexType output = (IndexType)threadIdx.x + (IndexType)blockIdx.x * (IndexType)blockDim.x;
    IndexType split  = blockIdx.y;
    if (output >= num_outputs) return;

    auto index_input      = a.LinearIndexToDimIndexSkipDim(output, dim);
    const InputType* base = &a[index_input];
    IndexType stride      = a.strides[dim];

    IndexType begin = split * split_length;
    IndexType end   = begin + split_length < size_to_reduce ? begin + split_length : size_to_reduce;
    AccType value   = default_value;
    for (IndexType k = begin; k < end; ++k)
    {
        value = op(value, op.load_op(AccType(base[k * stride])));
    }
    dimensional_reduce_store(result, partials, num_splits, output, split, value);
}

// One block per (output, split), grid-stride over the pairs. Used for long reductions over the innermost dimension:
// the threads of a block read adjacent elements. Every thread accumulates its elements in a register, the block is
// reduced once at the end.
template <typename InputType, typename AccType, typename Op, int REDUCE_BLOCK_SIZE, int MAX_DIMS, int REDUCE_DIM>
static __global__ void dimensional_reduce(TensorInfoCuda<InputType, MAX_DIMS> a, int64_t dynamic_dim,
                                          TensorInfoCuda<InputType, MAX_DIMS> result, AccType* partials,
                                          int64_t num_splits, Op op, AccType default_value)
{
    static_assert(REDUCE_BLOCK_SIZE % 32 == 0, "the block reduction requires full warps");
    int64_t dim = REDUCE_DIM == -1 ? dynamic_dim : REDUCE_DIM;

    using IndexType          = typename TensorInfoCuda<InputType, MAX_DIMS>::IndexType;
    IndexType size_to_reduce = a.size(dim);
    IndexType num_outputs    = a.numel() / size_to_reduce;
    IndexType split_length   = reduce_div_up(size_to_reduce, num_splits);
    IndexType num_work       = num_outputs * (IndexType)num_splits;
    IndexType stride         = a.strides[dim];

    for (IndexType work_id = blockIdx.x; work_id < num_work; work_id += gridDim.x)
    {
        IndexType output = work_id / (IndexType)num_splits;
        IndexType split  = work_id % (IndexType)num_splits;

        auto index_input      = a.LinearIndexToDimIndexSkipDim(output, dim);
        const InputType* base = &a[index_input];

        IndexType begin = split * split_length;
        IndexType end   = begin + split_length < size_to_reduce ? begin + split_length : size_to_reduce;
        AccType value   = default_value;
#pragma unroll 4
        for (IndexType k = begin + threadIdx.x; k < end; k += REDUCE_BLOCK_SIZE)
        {
            value = op(value, op.load_op(AccType(base[k * stride])));
        }
        // blockReduce synchronizes before it returns, so the next iteration can reuse its shared memory
        value = blockReduce<REDUCE_BLOCK_SIZE, AccType>(value, op, default_value);
        if (threadIdx.x == 0)
        {
            dimensional_reduce_store(result, partials, num_splits, output, split, value);
        }
    }
}

// Second pass of a split reduction: one block per output combines its num_splits partials (load_op was already
// applied to the elements).
template <typename InputType, typename AccType, typename Op, int REDUCE_BLOCK_SIZE, int MAX_DIMS>
static __global__ void dimensional_reduce_partials(const AccType* partials, int64_t num_outputs, int64_t num_splits,
                                                   TensorInfoCuda<InputType, MAX_DIMS> result, Op op,
                                                   AccType default_value)
{
    for (int64_t output = blockIdx.x; output < num_outputs; output += gridDim.x)
    {
        AccType value = default_value;
        for (int64_t k = threadIdx.x; k < num_splits; k += REDUCE_BLOCK_SIZE)
        {
            value = op(value, partials[output * num_splits + k]);
        }
        value = blockReduce<REDUCE_BLOCK_SIZE, AccType>(value, op, default_value);
        if (threadIdx.x == 0)
        {
            CUDA_KERNEL_ASSERT(output < result.numel());
            result[output] = InputType(value);
        }
    }
}

// The number of parts a reduction is split into so that about `target_work` units run in parallel, but every part
// keeps at least min_split_length elements.
static int64_t reduce_num_splits(int64_t num_outputs, int64_t size_to_reduce, int64_t target_work,
                                 int64_t min_split_length)
{
    int64_t wanted        = reduce_div_up(target_work, num_outputs);
    int64_t max_by_length = std::max<int64_t>(size_to_reduce / min_split_length, 1);
    return std::clamp<int64_t>(std::min(wanted, max_by_length), 1, REDUCE_MAX_SPLITS);
}

template <typename InputType, typename AccType, typename Op, int MAX_DIMS, int REDUCE_DIM>
void dimensional_reduce_launcher_size_dim(TensorInfoCuda<InputType, MAX_DIMS> a, int64_t dim,
                                          TensorInfoCuda<InputType, MAX_DIMS> result)
{
    int64_t size_to_reduce = a.size(dim);
    int64_t num_outputs    = a.numel() / size_to_reduce;

    CHECK_EQ(result.numel(), num_outputs);

    auto stream        = cuda::getCurrentCUDAStream();
    bool block_per_out = a.strides[dim] == 1 && size_to_reduce > 32;

    int64_t num_splits = block_per_out
                             ? reduce_num_splits(num_outputs, size_to_reduce, REDUCE_TARGET_BLOCKS, REDUCE_MIN_SPLIT_BLOCK)
                             : reduce_num_splits(num_outputs, size_to_reduce, REDUCE_TARGET_THREADS,
                                                 REDUCE_MIN_SPLIT_THREAD);

    Tensor partials;
    AccType* partials_ptr = nullptr;
    if (num_splits > 1)
    {
        partials =
            empty({num_outputs * num_splits},
                  TensorOptions().dtype(CppTypeToScalarType<AccType>::value).device(Device(kCUDA, cuda::getDevice())));
        partials_ptr = partials.template data_ptr<AccType>();
    }

    if (block_per_out)
    {
        int64_t num_blocks = std::min(num_outputs * num_splits, int64_t(1024) * 1024 / REDUCE_BLOCK_SIZE);
        dimensional_reduce<InputType, AccType, Op, REDUCE_BLOCK_SIZE, MAX_DIMS, REDUCE_DIM>
            <<<(unsigned int)num_blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(a, dim, result, partials_ptr, num_splits, Op(),
                                                                         Op::template default_value<AccType>());
    }
    else
    {
        // few outputs (e.g. 3 columns summed over a million rows) would leave most threads of a full block idle
        int64_t block_size = std::min<int64_t>(REDUCE_BLOCK_SIZE, iAlignUp(num_outputs, int64_t(32)));
        int64_t num_blocks = reduce_div_up(num_outputs, block_size);
        CHECK_LE(num_blocks, int64_t(std::numeric_limits<int32_t>::max()));
        dim3 grid((unsigned int)num_blocks, (unsigned int)num_splits);
        dimensional_reduce_thread<InputType, AccType, Op, MAX_DIMS, REDUCE_DIM>
            <<<grid, (unsigned int)block_size, 0, stream>>>(a, dim, result, partials_ptr, num_splits, Op(),
                                                            Op::template default_value<AccType>());
    }
    CUDA_SYNC_CHECK_ERROR();

    if (num_splits > 1)
    {
        int64_t num_blocks = std::min(num_outputs, int64_t(1024) * 1024 / REDUCE_BLOCK_SIZE);
        dimensional_reduce_partials<InputType, AccType, Op, REDUCE_BLOCK_SIZE, MAX_DIMS>
            <<<(unsigned int)num_blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(partials_ptr, num_outputs, num_splits, result,
                                                                         Op(), Op::template default_value<AccType>());
        CUDA_SYNC_CHECK_ERROR();
    }
}

template <typename InputType, typename AccType, typename Op, int MAX_DIMS>
void dimensional_reduce_launcher_size(Tensor a, int64_t dim, Tensor result)
{
    cuda::DeviceGuard guard(a.device());
    switch (dim)
    {
        case 0:
            dimensional_reduce_launcher_size_dim<InputType, AccType, Op, MAX_DIMS, 0>(a, dim, result);
            break;
        case 1:
            dimensional_reduce_launcher_size_dim<InputType, AccType, Op, MAX_DIMS, 1>(a, dim, result);
            break;
        default:
            dimensional_reduce_launcher_size_dim<InputType, AccType, Op, MAX_DIMS, -1>(a, dim, result);
            break;
    }
}

template <typename InputType, typename AccType, typename Op>
void dimensional_reduce_launcher(Tensor a, int64_t dim, Tensor result)
{
    if (a.dim() == 2 && result.dim() == 2)
    {
        dimensional_reduce_launcher_size<InputType, AccType, Op, 2>(a, dim, result);
    }
    else
    {
        dimensional_reduce_launcher_size<InputType, AccType, Op, -1>(a, dim, result);
    }
}


// half is accumulated in float and all integer types in int64
template <typename Op>
void dimensional_reduce_helper(Tensor a, int64_t dim, Tensor result)
{
    if (a.numel() == 0) return;
    cuda::DeviceGuard guard(a.device());
    switch (a.scalar_type())
    {
        case kUInt8:
            dimensional_reduce_launcher<uint8_t, int64_t, Op>(a, dim, result);
            break;
        case kInt16:
            dimensional_reduce_launcher<int16_t, int64_t, Op>(a, dim, result);
            break;
        case kUInt16:
            dimensional_reduce_launcher<uint16_t, int64_t, Op>(a, dim, result);
            break;
        case kInt32:
            dimensional_reduce_launcher<int32_t, int64_t, Op>(a, dim, result);
            break;
        case kInt64:
            dimensional_reduce_launcher<int64_t, int64_t, Op>(a, dim, result);
            break;
        case kHalf:
            dimensional_reduce_launcher<__half, float, Op>(a, dim, result);
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



template <typename InputType, typename AccType, typename Op>
static __global__ void dimensional_scan(TensorInfoCuda<InputType> a, int64_t dim, TensorInfoCuda<InputType> result,
                                        Op op, AccType default_value)
{
    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;
    int64_t num_steps            = iDivUp(size_to_reduce, REDUCE_BLOCK_SIZE);
    __shared__ AccType previous_block_sum;



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
            AccType local_value = i < size_to_reduce ? AccType(a[index_input]) : default_value;
            auto value          = blockInclusiveScan<REDUCE_BLOCK_SIZE, AccType, Op>(local_value, op, default_value);
            value               = op(value, previous_block_sum_copy);

            if (i < size_to_reduce)
            {
                result[index_input] = InputType(value);
            }

            if (threadIdx.x == REDUCE_BLOCK_SIZE - 1)
            {
                previous_block_sum = value;
            }
            __syncthreads();
        }
    }
}

template <typename InputType, typename AccType, typename Op>
void dimensional_scan_launcher(TensorInfoCuda<InputType> a, int64_t dim, TensorInfoCuda<InputType> result)
{
    int64_t size_to_reduce       = a.size(dim);
    int64_t num_blocks_to_reduce = a.numel() / size_to_reduce;
    int64_t num_blocks           = std::min(num_blocks_to_reduce, int64_t(1024) * 1024 / REDUCE_BLOCK_SIZE);

    dimensional_scan<InputType, AccType, Op>
        <<<(unsigned int)num_blocks, REDUCE_BLOCK_SIZE, 0, cuda::getCurrentCUDAStream()>>>(
            a, dim, result, Op(), Op::template default_value<AccType>());
    CUDA_SYNC_CHECK_ERROR();
}

// half is accumulated in float and all integer types in int64
template <typename Op>
void dimensional_scan_helper(Tensor a, int64_t dim, Tensor result)
{
    if (a.numel() == 0) return;
    cuda::DeviceGuard guard(a.device());
    switch (a.scalar_type())
    {
        case kUInt8:
            dimensional_scan_launcher<uint8_t, int64_t, Op>(a, dim, result);
            break;
        case kInt16:
            dimensional_scan_launcher<int16_t, int64_t, Op>(a, dim, result);
            break;
        case kUInt16:
            dimensional_scan_launcher<uint16_t, int64_t, Op>(a, dim, result);
            break;
        case kInt32:
            dimensional_scan_launcher<int32_t, int64_t, Op>(a, dim, result);
            break;
        case kInt64:
            dimensional_scan_launcher<int64_t, int64_t, Op>(a, dim, result);
            break;
        case kHalf:
            dimensional_scan_launcher<__half, float, Op>(a, dim, result);
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
