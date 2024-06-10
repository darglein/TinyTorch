/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/tensor.h"

#include "torch/core/tensor_options.h"
#include "torch/tiny_torch_config.h"
#include "torch/tiny_torch_cuda.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


template <int Size>
struct AtomicCASType
{
};

template <>
struct AtomicCASType<2>
{
    using Type = unsigned short int;
};
template <>
struct AtomicCASType<4>
{
    using Type = int;
};
template <>
struct AtomicCASType<8>
{
    using Type = unsigned long long int;
};


template <typename T, typename Op>
__device__ static T atomic_op_with_cas(T* address, Op op)
{
    using CAS_TYPE = typename AtomicCASType<sizeof(T)>::Type;

    CAS_TYPE* address_as_i = (CAS_TYPE*)address;
    CAS_TYPE old           = *address_as_i, assumed;
    do
    {
        assumed                = old;
        T assumed_float        = ((T*)&assumed)[0];
        T new_value            = op(assumed_float);
        CAS_TYPE new_value_int = ((CAS_TYPE*)&new_value)[0];

        old = ::atomicCAS(address_as_i, assumed, new_value_int);
    } while (assumed != old);
    T old_value = ((T*)&old)[0];
    return old_value;
}

template <typename T>
__device__ static T atomicMulSelect(T* address, T a)
{
    return atomic_op_with_cas(address, [a](auto b) { return a * b; });
}

template <typename T>
__device__ static T atomicMinSelect(T* address, T a)
{
    return atomic_op_with_cas(address, [a](auto b) { return a < b ? a : b; });
}

template <typename T>
__device__ static T atomicMaxSelect(T* address, T a)
{
    return atomic_op_with_cas(address, [a](auto b) { return a > b ? a : b; });
}


template <typename T>
__device__ inline T atomicAddSelect(T* address, T val)
{
    return atomicAdd(address, val);
}
template <>
__device__ inline uint16_t atomicAddSelect(uint16_t* address, uint16_t val)
{
    return atomic_op_with_cas(address, [val](auto a) { return a + val; });
}

template <>
__device__ inline int64_t atomicAddSelect(int64_t* address, int64_t a)
{
    return atomic_op_with_cas(address, [a](auto b) { return a + b; });
}
