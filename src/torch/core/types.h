/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "assert.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "torch/tiny_torch_config.h"

#if defined(__CUDACC__)
#include "cuda_fp16.h"
#endif

namespace tinytorch
{

struct Half
{
};
enum Device
{
    kCPU,
    kCUDA
};

enum ScalarType
{
    // interger types
    kUInt8,
    kByte = kUInt8,
    kInt16,
    kShort = kInt16,
    kInt32,
    kInt = kInt32,
    kInt64,
    kLong = kInt64,
    // floating point types
    kFloat16,
    kHalf = kFloat16,
    kFloat32,
    kFloat = kFloat32,
    kFloat64,
    kDouble = kFloat64,
};

template <typename T>
struct CppTypeToScalarType
{
};
#if defined(__CUDACC__)
template <>
struct CppTypeToScalarType<__half>
{
    static constexpr ScalarType value = kFloat16;
};
#endif
template <>
struct CppTypeToScalarType<float>
{
    static constexpr ScalarType value = kFloat32;
};
template <>
struct CppTypeToScalarType<double>
{
    static constexpr ScalarType value = kFloat64;
};
using Dtype = ScalarType;


template <typename T>
ScalarType dtype()
{
    return CppTypeToScalarType<T>::value;
}

}  // namespace tinytorch
