#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/tiny_torch_cuda.h"

#include <cuda_runtime.h>

TT_HD constexpr uint32_t iDivUp(int64_t a, int64_t b)
{
    return (uint32_t)((a + b - 1) / b);
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace tinytorch
{
#define CASE_MACRO(func, type, scalar_type, numel, ...)                  \
    case scalar_type:                                                    \
        func<type><<<iDivUp(numel, 128), 128, 0, stream>>>(__VA_ARGS__); \
        break;

#define SWITCH_MACRO_FLOAT(real_scalar_type, numel, func, ...)         \
    auto stream = cuda::getCurrentCUDAStream();                        \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, float, kFloat, numel, __VA_ARGS__)            \
        CASE_MACRO(func, double, kDouble, numel, __VA_ARGS__)          \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }

// TODO: Half!
#define SWITCH_MACRO_ALL(real_scalar_type, numel, func, ...)           \
    auto stream = cuda::getCurrentCUDAStream();                        \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, uint8_t, kUInt8, numel, __VA_ARGS__)          \
        CASE_MACRO(func, int16_t, kInt16, numel, __VA_ARGS__)          \
        CASE_MACRO(func, int32_t, kInt32, numel, __VA_ARGS__)          \
        CASE_MACRO(func, int64_t, kLong, numel, __VA_ARGS__)           \
        CASE_MACRO(func, float, kFloat, numel, __VA_ARGS__)            \
        CASE_MACRO(func, double, kDouble, numel, __VA_ARGS__)          \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }



template <typename T>
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

template <typename T>
static __global__ void copy_impl_cuda(TensorInfo<T> a, TensorInfo<T> b)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    b[i] = a[i];
}
void copy_impl_cuda(Tensor src, Tensor target)
{
    SWITCH_MACRO_ALL(src.scalar_type(), src.numel(), copy_impl_cuda, src, target);
}


template <typename T>
static __global__ void square_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    auto v    = a[i];
    result[i] = v * v;
}

Tensor square_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), square_impl_cuda, a, result);
    return result;
}


template <typename T>
static __global__ void sum_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    T* ptr = &result[0];
    atomicAdd(ptr, a[i]);
}

Tensor sum_impl_cuda(Tensor a)
{
    Tensor result = zeros({1}, a.options().requires_grad(false));

    auto stream = cuda::getCurrentCUDAStream();
    switch (a.scalar_type())
    {
        CASE_MACRO(sum_impl_cuda, int32_t, kInt32, a.numel(), a, result)
        CASE_MACRO(sum_impl_cuda, float, kFloat, a.numel(), a, result)
        CASE_MACRO(sum_impl_cuda, double, kDouble, a.numel(), a, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }

    return result;
}

template <typename T>
static __global__ void log_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::log(a[i]);
}

Tensor log_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), log_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void log1p_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::log1p(a[i]);
}

Tensor log1p_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), log1p_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void exp_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::exp(a[i]);
}

Tensor exp_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), exp_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void sign_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    T v       = a[i];
    result[i] = (v < T(0)) ? T(-1) : (v > T(0)) ? T(1) : T(0);
}

Tensor sign_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sign_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void pow_impl_cuda(TensorInfo<T> a, double b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = T(std::pow(a[i], b));
}

Tensor pow_impl_cuda(Tensor a, double b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), pow_impl_cuda, a, b, result);
    return result;
}

template <typename T>
static __global__ void sin_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::sin(a[i]);
}

Tensor sin_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sin_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void cos_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = std::cos(a[i]);
}

Tensor cos_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), cos_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void relu_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = relu(a[i]);
}

Tensor relu_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), relu_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void sigmoid_impl_cuda(TensorInfo<T> a, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = sigmoid(a[i]);
}

Tensor sigmoid_impl_cuda(Tensor a)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), sigmoid_impl_cuda, a, result);
    return result;
}

template <typename T>
static __global__ void softplus_impl_cuda(TensorInfo<T> a, double beta, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = softplus(a[i], T(beta));
}

Tensor softplus_impl_cuda(Tensor a, double beta)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), softplus_impl_cuda, a, beta, result);
    return result;
}

template <typename T>
static __global__ void min_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MIN(a[i], b[i]);
}

Tensor min_impl_cuda(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), min_impl_cuda, a, b, result);
    return result;
}

template <typename T>
static __global__ void max_impl_cuda(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MAX(a[i], b[i]);
}

Tensor max_impl_cuda(Tensor a, Tensor b)
{
    Tensor result = empty_like(a);
    SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), max_impl_cuda, a, b, result);
    return result;
}






}  // namespace tinytorch
