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

#include "tiny_torch_config.h"

namespace tinytorch
{

enum DeviceType
{
    kCPU,
    kCUDA
};

enum ScalarType
{
    kFloat,
};

struct StorageImpl
{
    StorageImpl(int64_t size, DeviceType device);

    StorageImpl& operator=(StorageImpl&& other) = default;
    StorageImpl& operator=(const StorageImpl&)  = delete;
    StorageImpl()                               = delete;
    StorageImpl(StorageImpl&& other)            = default;
    StorageImpl(const StorageImpl&)             = delete;
    ~StorageImpl();

    uint8_t* byte_ptr() { return (uint8_t*)data_ptr_; }

   protected:
    DeviceType device_;
    void* data_ptr_ = nullptr;
    int64_t size_;
};



}  // namespace tinytorch
