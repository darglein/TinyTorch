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

namespace TINY_TORCH_NAMESPACE
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
    kByte,
    kUInt8 = kByte,
    kInt16,
    kShort = kInt16,
    kInt32,
    kInt = kInt32,
    kLong,
    kFloat32,
    kFloat = kFloat32,
    kFloat64,
    kDouble = kFloat64,
    kHalf,
};

using Dtype = ScalarType;

struct StorageImpl
{
    StorageImpl(int64_t size, Device device);

    StorageImpl& operator=(StorageImpl&& other) = default;
    StorageImpl& operator=(const StorageImpl&)  = delete;
    StorageImpl()                               = delete;
    StorageImpl(StorageImpl&& other)            = default;
    StorageImpl(const StorageImpl&)             = delete;
    ~StorageImpl();

    uint8_t* byte_ptr() { return (uint8_t*)data_ptr_; }

   protected:
    Device device_;
    void* data_ptr_ = nullptr;
    int64_t size_;
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
            assert(false);
    }
    return 0;
}



}  // namespace TINY_TORCH_NAMESPACE
