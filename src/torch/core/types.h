/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "assert.h"
#include "torch/core/half.h"

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


enum DeviceType
{
    kCPU,
    kCUDA
};


struct TINYTORCH_API Device
{
    DeviceType _type  = kCPU;
    int device_index = 0;

    Device(DeviceType type = kCPU, int index = 0) : _type(type), device_index(index)
    {
        if (type == kCPU)
        {
            CHECK_EQ(index, 0);
        }
    }

     DeviceType type(){ return _type; }
};

inline bool operator==(Device device, DeviceType type)
{
    return device.type() == type;
}
inline bool operator==(Device d1, Device d2)
{
    return d1.type() == d2.type() && d1.device_index == d2.device_index;
}


TINYTORCH_API std::ostream& operator<<(std::ostream& strm, Device type);

enum PaddingMode
{
    kBorder,
};
enum InterpolationType
{
    kBilinear,
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

TINYTORCH_API std::ostream& operator<<(std::ostream& strm, ScalarType type);

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
        case kInt64:
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
struct CppTypeToScalarType<uint8_t>
{
    static constexpr ScalarType value = kUInt8;
};
template <>
struct CppTypeToScalarType<int16_t>
{
    static constexpr ScalarType value = kInt16;
};
template <>
struct CppTypeToScalarType<int32_t>
{
    static constexpr ScalarType value = kInt32;
};
template <>
struct CppTypeToScalarType<int64_t>
{
    static constexpr ScalarType value = kInt64;
};
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
