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
#    include "cuda_fp16.h"
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
    kUInt8 = 0,
    kByte  = kUInt8,
    kInt16 = 1,
    kShort = kInt16,
    kInt32 = 2,
    kInt   = kInt32,
    kInt64 = 3,
    kLong  = kInt64,
    // floating point types
    kFloat16 = 4,
    kHalf    = kFloat16,
    kFloat32 = 5,
    kFloat   = kFloat32,
    kFloat64 = 6,
    kDouble  = kFloat64,
};

inline int64_t elementSize(ScalarType type)
{
    switch (type)
    {
        case kUInt8:
            return 1;
        case kInt16:
            return 2;
        case kInt32:
            return 4;
        case kLong:
            return 8;
        case kFloat32:
            return 4;
        case kFloat64:
            return 8;
        case kHalf:
            return 2;
        default:
            CHECK(false);
    }
    return 0;
}


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

inline std::ostream& operator<<(std::ostream& strm, tinytorch::Device d)
{
    strm << ((d == tinytorch::kCPU) ? "kCPU" : "kCUDA");
    return strm;
}

inline std::ostream& operator<<(std::ostream& strm, tinytorch::ScalarType type)
{
    switch (type)
    {
        case tinytorch::kUInt8:
            strm << "kUint8";
            break;
        case tinytorch::kInt16:
            strm << "kInt16";
            break;
        case tinytorch::kInt32:
            strm << "kInt32";
            break;
        case tinytorch::kLong:
            strm << "kLong";
            break;
        case tinytorch::kFloat32:
            strm << "kFloat32";
            break;
        case tinytorch::kFloat64:
            strm << "kFloat64";
            break;
        case tinytorch::kHalf:
            strm << "kHalf";
            break;
        default:
            CHECK(false);
    }
    return strm;
}
