#pragma once

#include "torch/cpu/binary_operators.h"
#include "torch/cpu/ops_impl_cpu.h"
#include "torch/cpu/unary_operators.h"
#include "torch/cuda/binary_operators.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/unary_operators.h"
#include "torch/tiny_torch_config.h"


#ifdef TT_HAS_CUDA
#    define SELECT_DEVICE_CUDA(func, ...) \
        case kCUDA:                       \
            cuda_impl::func(__VA_ARGS__); \
            break;
#else
#    define SELECT_DEVICE_CUDA(...)
#endif


#define SELECT_DEVICE(device_type, func, ...)                     \
    switch (device_type.type)                                     \
    {                                                             \
        case kCPU:                                                \
            cpu_impl::func(__VA_ARGS__);                          \
            break;                                                \
            SELECT_DEVICE_CUDA(func, __VA_ARGS__)                 \
        default:                                                  \
            CHECK(false) << "invalid input type " << device_type; \
    }
