#pragma once


#include "torch/tiny_torch_cuda.h"

#include <cuda_fp16.h>

TT_HD constexpr uint32_t iDivUp(int64_t a, int64_t b)
{
    return (uint32_t)((a + b - 1) / b);
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef __launch_bounds__
#    define __launch_bounds__(...)
#endif

#define CASE_MACRO(func, type, scalar_type, numel, ...)                  \
    case scalar_type:                                                    \
        func<type><<<iDivUp(numel, 128), 128, 0, stream>>>(__VA_ARGS__); \
        break;

#define SWITCH_MACRO_FLOAT(real_scalar_type, numel, func, ...)         \
    auto stream = cuda::getCurrentCUDAStream();                        \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, half, kHalf, numel, __VA_ARGS__)              \
        CASE_MACRO(func, float, kFloat, numel, __VA_ARGS__)            \
        CASE_MACRO(func, double, kDouble, numel, __VA_ARGS__)          \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }

#define SWITCH_MACRO_ALL(real_scalar_type, numel, func, ...)           \
    auto stream = cuda::getCurrentCUDAStream();                        \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, uint8_t, kUInt8, numel, __VA_ARGS__)          \
        CASE_MACRO(func, int16_t, kInt16, numel, __VA_ARGS__)          \
        CASE_MACRO(func, int32_t, kInt32, numel, __VA_ARGS__)          \
        CASE_MACRO(func, int64_t, kLong, numel, __VA_ARGS__)           \
        CASE_MACRO(func, half, kHalf, numel, __VA_ARGS__)              \
        CASE_MACRO(func, float, kFloat, numel, __VA_ARGS__)            \
        CASE_MACRO(func, double, kDouble, numel, __VA_ARGS__)          \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }


#ifdef __CUDACC__

inline __device__ __half sqrt(__half a)
{
    return hsqrt(a);
}

inline __device__ __half exp(__half a)
{
    return hexp(a);
}

inline __device__ __half log(__half a)
{
    return hlog(a);
}

inline __device__ __half log1p(__half a)
{
    return hlog(__half(1.f) + a);
}

inline __device__ __half sin(__half a)
{
    return hsin(a);
}

inline __device__ __half cos(__half a)
{
    return hcos(a);
}

inline __device__ __half pow(__half a, double b)
{
    return __float2half(pow(__half2float(a), b));
}

#endif

