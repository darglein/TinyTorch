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
    CUDA_SWITCH_MACRO_FLOAT(input.device(), input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, true);
    indices = Tensor();
}

void max_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    CUDA_SWITCH_MACRO_FLOAT(input.device(), input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, false);
    indices = Tensor();
}
}  // namespace cuda_impl
}  // namespace tinytorch