#pragma once


#include "torch/core/tensor_info.h"
#include "torch/tiny_torch_cuda.h"

TT_HD constexpr uint32_t iDivUp(int64_t a, int64_t b)
{
    return (uint32_t)((a + b - 1) / b);
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef __launch_bounds__
#    define __launch_bounds__(...)
#endif

#if !defined(CUDA_NDEBUG)
#    if !defined(CUDA_DEBUG)
#        define CUDA_DEBUG
#    endif
#else
#    undef CUDA_DEBUG
#endif

#define CHECK_CUDA_ERROR(cudaFunction)                                                        \
    {                                                                                         \
        cudaError_t cudaErrorCode = cudaFunction;                                             \
        CHECK_EQ(cudaErrorCode, cudaSuccess)                                                  \
            << ": " << cudaGetErrorString(cudaErrorCode) << " in function " << #cudaFunction; \
    }

#if defined(CUDA_DEBUG) && TT_DEBUG
#    define CUDA_SYNC_CHECK_ERROR()                    \
        {                                              \
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); \
        }
#else
#    define CUDA_SYNC_CHECK_ERROR() (static_cast<void>(0))
#endif

#define EXPAND(x) x  // https://stackoverflow.com/questions/32399191/va-args-expansion-using-msvc

#define CUDA_CASE_MACRO_REFINED(block_size, func, scalar_type, numel, ...)                                 \
    case scalar_type:                                                                                      \
        if (numel > 0)                                                                                     \
        {                                                                                                  \
            func<<<iDivUp(numel, block_size), block_size, 0, cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); \
            CUDA_SYNC_CHECK_ERROR();                                                                       \
        }                                                                                                  \
        break;


#define CUDA_CASE_MACRO(...) EXPAND(CUDA_CASE_MACRO_REFINED(128, __VA_ARGS__))

#define CUDA_SWITCH_MACRO_FLOAT(real_scalar_type, numel, func, ...)        \
    {                                                                      \
        switch (real_scalar_type)                                          \
        {                                                                  \
            CUDA_CASE_MACRO(func<half>, kHalf, numel, __VA_ARGS__)         \
            CUDA_CASE_MACRO(func<float>, kFloat, numel, __VA_ARGS__)       \
            CUDA_CASE_MACRO(func<double>, kDouble, numel, __VA_ARGS__)     \
            default:                                                       \
                CHECK(false) << "invalid input type " << real_scalar_type; \
        }                                                                  \
    }

// TODO: Half!
#define CUDA_SWITCH_MACRO_ALL(real_scalar_type, numel, func, ...)      \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CUDA_CASE_MACRO(func<uint8_t>, kUInt8, numel, __VA_ARGS__)     \
        CUDA_CASE_MACRO(func<int16_t>, kInt16, numel, __VA_ARGS__)     \
        CUDA_CASE_MACRO(func<int32_t>, kInt32, numel, __VA_ARGS__)     \
        CUDA_CASE_MACRO(func<int64_t>, kLong, numel, __VA_ARGS__)      \
        CUDA_CASE_MACRO(func<half>, kHalf, numel, __VA_ARGS__)         \
        CUDA_CASE_MACRO(func<float>, kFloat, numel, __VA_ARGS__)       \
        CUDA_CASE_MACRO(func<double>, kDouble, numel, __VA_ARGS__)     \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }


// used for conversion
#define CUDA_SWITCH_MACRO_ALL_DUAL(real_scalar_type, second_type, numel, func, ...) \
    switch (real_scalar_type)                                                       \
    {                                                                               \
        CUDA_CASE_MACRO((func<uint8_t, second_type>), kUInt8, numel, __VA_ARGS__)   \
        CUDA_CASE_MACRO((func<int16_t, second_type>), kInt16, numel, __VA_ARGS__)   \
        CUDA_CASE_MACRO((func<int32_t, second_type>), kInt32, numel, __VA_ARGS__)   \
        CUDA_CASE_MACRO((func<int64_t, second_type>), kLong, numel, __VA_ARGS__)    \
        CUDA_CASE_MACRO((func<half, second_type>), kHalf, numel, __VA_ARGS__)       \
        CUDA_CASE_MACRO((func<float, second_type>), kFloat, numel, __VA_ARGS__)     \
        CUDA_CASE_MACRO((func<double, second_type>), kDouble, numel, __VA_ARGS__)   \
        default:                                                                    \
            CHECK(false) << "invalid input type " << real_scalar_type;              \
    }
