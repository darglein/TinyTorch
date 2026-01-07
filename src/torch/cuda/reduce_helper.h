/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

#include "torch/core/tensor_options.h"
#include "torch/cuda/atomic_minmax.h"
#include "torch/tiny_torch_config.h"
namespace tinytorch
{
namespace cuda_impl
{
constexpr int REDUCE_BLOCK_SIZE = 256;


struct ReduceAdd
{
    template <typename T>
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a + b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicAddSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return T(0);
    }
};

struct ReduceAbsAdd
{
    template <typename T>
    TT_HD T load_op(T a)
    {
        return a > T(0) ? a : -a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a + b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicAddSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return T(0);
    }
};

struct ReduceProdAdd
{
    template <typename T>
    TT_HD T load_op(T a)
    {
        return a * a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a + b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicAddSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return T(0);
    }
};

struct ReduceProd
{
    template <typename T>
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a * b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicMulSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return T(1);
    }
};
struct ReduceMin
{
    template <typename T>
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a < b ? a : b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicMinSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return std::numeric_limits<T>::max();
    }
};
struct ReduceMax
{
    template <typename T>
    TT_HD T load_op(T a)
    {
        return a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a > b ? a : b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicMaxSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return std::numeric_limits<T>::lowest();
    }
};
struct StdHelper
{
    StdHelper(void* mean_ptr) : mean_ptr(mean_ptr) {}
    template <typename T>
    TT_HD T load_op(T a)
    {
        a = a - ((T*)mean_ptr)[0];
        return a * a;
    }

    template <typename T>
    TT_HD T operator()(T a, T b)
    {
        return a + b;
    }
    template <typename T>
    TT_HD T atomic_reduce(T* target, T value)
    {
        return atomicAddSelect(target, value);
    }
    template <typename T>
    static constexpr T default_value()
    {
        return 0;
    }

    void* mean_ptr;
};


template <typename T, typename ShuffleType = int>
__device__ inline T shfl_xor(T var, unsigned int srcLane, int width = 32)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_xor_sync(0xFFFFFFFF, a[i], srcLane, width);
    }
    return var;
}

template <typename T, typename ShuffleType = int>
__device__ inline T shfl_up(T var, unsigned int srcLane, int width = 32)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_up_sync(0xFFFFFFFF, a[i], srcLane, width);
    }
    return var;
}

template <typename T, typename OP, unsigned int LOCAL_WARP_SIZE = 32>
__device__ inline T warpReduce(T val, OP op)
{
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = shfl_xor<T, T>(val, offset);
        val    = op(val, v);
    }
    return val;
}


template <int BLOCK_SIZE, typename T, typename OP>
__device__ inline T blockReduce(T val, OP op, T default_val)
{
    __shared__ T shared[BLOCK_SIZE / 32];

    int lane   = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;

    // Each warp reduces with registers
    val = warpReduce(val, op);

    // The first thread in each warp writes to smem
    if (lane == 0)
    {
        shared[warpid] = val;
    }

    __syncthreads();

    if (warpid == 0)
    {
        if (threadIdx.x < BLOCK_SIZE / 32)
        {
            val = shared[threadIdx.x];
        }
        else
        {
            val = default_val;
        }


        val = warpReduce(val, op);
    }
    __syncthreads();

    return val;
}

}  // namespace cuda_impl
}  // namespace tinytorch