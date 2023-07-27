#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/binary_operators.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>

namespace tinytorch
{
namespace cuda_impl
{
template <typename T>
__launch_bounds__(128) static __global__ void add_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] + b.data[offset_b];
}

void add_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), add_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void add_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = a[i] + T(b);
}

void add_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), add_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void sub_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] - b.data[offset_b];
}

void sub_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), sub_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void sub_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = a[i] - T(b);
}

void sub_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), sub_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void mult_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] * b.data[offset_b];
}

void mult_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), mult_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void mult_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = a[i] * T(b);
}

void mult_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), mult_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void div_impl(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] / b.data[offset_b];
}

void div_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), div_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void div_impl(double a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a / double(b[i]));
}

void div_impl(double a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(b.scalar_type(), b.numel(), div_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void equal_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a[i] == T(b));
}

void equal_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), equal_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void less_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a[i] < T(b));
}

void less_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), less_impl, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void greater_impl(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a[i] > T(b));
}

void greater_impl(Tensor a, double b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), greater_impl, a, b, result);
}

}  // namespace tinytorch
}