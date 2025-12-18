
#include "torch/core/ops/ops_impl.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{
namespace cuda_impl
{


template <typename T, int MAX_SIZE, typename TIndex>
__launch_bounds__(128) static __global__
    void index_select_impl(TensorInfoCuda<T, MAX_SIZE> input, int64_t dim, TensorInfoCuda<TIndex, 1> index,
                           TensorInfoCuda<T, MAX_SIZE> result)
{
    int64_t result_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (result_linear_index >= result.numel()) return;

    auto index_result = result.LinearIndexToDimIndex(result_linear_index);
    auto index_input  = index_result;

    index_input.set_index(dim, index[index_result.get_index(dim)]);
    result[index_result] = input[index_input];
}

template <typename TIndex, int MAX_SIZE>
static void index_select_helper_dim(Tensor input, int64_t dim, TensorInfoCuda<TIndex, 1> index, Tensor result)
{
    CUDA_SWITCH_MACRO_ALL_DUAL(result.device(), result.scalar_type(), MAX_SIZE, result.numel(), index_select_impl,
                               input, dim, index, result);
}

template <typename TIndex>
static void index_select_helper(Tensor input, int64_t dim, TensorInfoCuda<TIndex, 1> index, Tensor result)
{
    CHECK_EQ(input.dim(), result.dim());
    switch (input.dim())
    {
        case 1:
            index_select_helper_dim<TIndex, 1>(input, dim, index, result);
            break;
        case 2:
            index_select_helper_dim<TIndex, 2>(input, dim, index, result);
            break;
        default:
            index_select_helper_dim<TIndex, -1>(input, dim, index, result);
            break;
    }
}

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_select_helper<int32_t>(input, dim, index, result);
            break;
        case kLong:
            index_select_helper<int64_t>(input, dim, index, result);
            break;
        default:
            CHECK(false) << "invalid input type " << index.scalar_type();
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
    index_result.set_index(dim, index[index_input.get_index(dim)]);


    CUDA_KERNEL_ASSERT(index_input[dim] < index.sizes[0]);
    CUDA_KERNEL_ASSERT(index_result[dim] < result.sizes[dim]);
    CUDA_KERNEL_ASSERT(result.index_in_range(index_result));
    CUDA_KERNEL_ASSERT(data.index_in_range(index_input));


    atomicAdd(&result[index_result], data[index_input]);
}

template <typename TIndex>
static void index_add_helper(Tensor data, int64_t dim, TensorInfoCuda<TIndex> index, Tensor result)
{
    CUDA_SWITCH_MACRO_FLOAT(result.device(), result.scalar_type(), data.numel(), index_add_impl, dim, index, data,
                            result);
}

void index_add_impl(int64_t dim, Tensor index, Tensor data, Tensor result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_add_helper<int32_t>(data, dim, index, result);
            break;
        case kLong:
            index_add_helper<int64_t>(data, dim, index, result);
            break;
        default:
            CHECK(false) << "invalid input type " << index.scalar_type();
    }
}


template <typename T>
__launch_bounds__(128) static __global__
    void gather_impl(TensorInfoCuda<T> data, int64_t dim, TensorInfoCuda<int64_t> index, TensorInfoCuda<T> result)
{
    int64_t input_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (input_linear_index >= result.numel()) return;

    {
        auto index_result = result.LinearIndexToDimIndex(input_linear_index);
        auto index_input  = index_result;

        // index_input[dim] = index[index_result];
        index_input.set_index(dim, index[index_result]);

        result[index_result] = data[index_input];
    }
}
void gather_impl(Tensor data, int64_t dim, Tensor index, Tensor result)
{
    CUDA_SWITCH_MACRO_ALL(data.device(), data.scalar_type(), result.numel(), gather_impl, data, dim, index, result);
}


template <typename T, typename TIndex>
__launch_bounds__(128) static __global__
    void index_copy_impl(TensorInfoCuda<T> target, int64_t dim, TensorInfoCuda<TIndex> index, TensorInfoCuda<T> value)
{
    int64_t input_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (input_linear_index >= value.numel()) return;

    auto index_input  = value.LinearIndexToDimIndex(input_linear_index);
    auto index_result = index_input;

    // index_result[dim] = index[index_input[dim]];
    index_result.set_index(dim, index[index_input.get_index(dim)]);


    CUDA_KERNEL_ASSERT(index_input[dim] < index.sizes[0]);
    CUDA_KERNEL_ASSERT(index_result[dim] < target.sizes[dim]);
    CUDA_KERNEL_ASSERT(target.index_in_range(index_result));
    CUDA_KERNEL_ASSERT(value.index_in_range(index_input));

    target[index_result] = value[index_input];
}

template <typename TIndex>
static void index_copy_helper(Tensor& target, int64_t dim, TensorInfoCuda<TIndex> index, Tensor value)
{
    CUDA_SWITCH_MACRO_ALL(target.device(), target.scalar_type(), value.numel(), index_copy_impl, target, dim, index,
                          value);
}

void index_copy_impl(Tensor& target, int64_t dim, Tensor index, Tensor value)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_copy_helper<int32_t>(target, dim, index, value);
            break;
        case kLong:
            index_copy_helper<int64_t>(target, dim, index, value);
            break;
        default:
            throw std::runtime_error("invalid type");
    }
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
void transpose_impl(Tensor input, int64_t dim0, int64_t dim1, Tensor result)
{
    CUDA_SWITCH_MACRO_ALL(result.device(), result.scalar_type(), result.numel(), transpose_impl, input, dim0, dim1,
                          result);
}


}  // namespace cuda_impl
}  // namespace tinytorch