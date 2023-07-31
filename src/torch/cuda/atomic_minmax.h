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
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace tinytorch
{
__device__ static half atomicMin(half* address, half val)
{
    CUDA_KERNEL_ASSERT(false);
    return val;
}
__device__ static half atomicMax(half* address, half val)
{
    CUDA_KERNEL_ASSERT(false);
    return val;
}
__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old           = *address_as_i, assumed;
    do
    {
        assumed = old;
        old     = ::atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ static double atomicMin(double* address, double val)
{
    using T = unsigned long long int;
    static_assert(sizeof(T) == sizeof(double), "match");
    T* address_as_i = (T*)address;
    T old           = *address_as_i, assumed;
    do
    {
        assumed = old;

        double assumed_float = ((double*)&assumed)[0];
        double new_value     = ::fmin(val, assumed_float);
        T new_value_int      = ((T*)&new_value)[0];

        old = ::atomicCAS(address_as_i, assumed, new_value_int);
    } while (assumed != old);
    double old_value = ((double*)&old)[0];
    return old_value;
}


__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old           = *address_as_i, assumed;
    do
    {
        assumed = old;
        old     = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ static double atomicMax(double* address, double val)
{
    using T = unsigned long long int;
    static_assert(sizeof(T) == sizeof(double), "match");
    T* address_as_i = (T*)address;
    T old           = *address_as_i, assumed;
    do
    {
        assumed = old;

        double assumed_float = ((double*)&assumed)[0];
        double new_value     = ::fmax(val, assumed_float);
        T new_value_int      = ((T*)&new_value)[0];

        old = ::atomicCAS(address_as_i, assumed, new_value_int);
    } while (assumed != old);
    double old_value = ((double*)&old)[0];
    return old_value;
}


}  // namespace tinytorch