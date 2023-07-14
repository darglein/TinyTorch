#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include "torch/core/ops_functions.h"
#include "torch/core/ops_impl_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/tiny_torch_cuda.h"

TT_HD constexpr uint32_t iDivUp(int64_t a, int64_t b)
{
    return (uint32_t)((a + b - 1) / b);
}

namespace tinytorch
{
#define CASE_MACRO(func, type, scalar_type, numel, ...)                  \
    case scalar_type:                                                    \
        func<type><<<iDivUp(numel, 128), 128, 0, stream>>>(__VA_ARGS__); \
        break;

#define SWITCH_MACRO_FLOAT(real_scalar_type, numel, func, ...)         \
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
    int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
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
    int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
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
    int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= a.numel()) return;
    b[i] = a[i];
}
void copy_impl_cuda(Tensor src, Tensor target)
{
    SWITCH_MACRO_ALL(src.scalar_type(), src.numel(), copy_impl_cuda, src, target);
}


}  // namespace tinytorch
