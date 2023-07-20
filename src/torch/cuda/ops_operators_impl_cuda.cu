#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/ops_operators_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"

#include <cuda_runtime.h>

namespace tinytorch
{

template <typename T>
static void print_impl_cpu(std::ostream& strm, TensorInfo<T> a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        strm << a[i] << " ";
    }
}

void print_impl_cuda(std::ostream& strm, Tensor t) 
{
    t = t.cpu();
    print_impl_cpu<float>(strm, t);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void add_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] + b.data[offset_b];
}

void add_impl_cuda(Tensor a, Tensor b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), add_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void add_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(double(a[i]) + b);
}

void add_impl_cuda(Tensor a, double b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), add_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void sub_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] - b.data[offset_b];
}

void sub_impl_cuda(Tensor a, Tensor b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), sub_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) static __global__ void sub_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(double(a[i]) - b);
}

void sub_impl_cuda(Tensor a, double b, Tensor& result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), sub_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void mult_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] * b.data[offset_b];
}

void mult_impl_cuda(Tensor a, Tensor b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), mult_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void mult_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(double(a[i]) * b);
}

void mult_impl_cuda(Tensor a, double b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), mult_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void div_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    int64_t dims = result.dims;

    int64_t offset_a, offset_b;
    calculate_offsets(i, dims, a.sizes, b.sizes, a.strides, b.strides, offset_a, offset_b);
    result[i] = a.data[offset_a] / b.data[offset_b];
}

void div_impl_cuda(Tensor a, Tensor b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), div_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void div_impl_cuda(double a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a / double(b[i]));
}

void div_impl_cuda(double a, Tensor b, Tensor& result) 
{
    SWITCH_MACRO_ALL(b.scalar_type(), b.numel(), div_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void equal_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a[i] == T(b));
}

void equal_impl_cuda(Tensor a, double b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), equal_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void less_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a[i] < T(b));
}

void less_impl_cuda(Tensor a, double b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), less_impl_cuda, a, b, result);
}

template <typename T>
__launch_bounds__(128) 
static __global__ void greater_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i] = T(a[i] > T(b));
}

void greater_impl_cuda(Tensor a, double b, Tensor& result) 
{
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), greater_impl_cuda, a, b, result);
}

}  // namespace tinytorch