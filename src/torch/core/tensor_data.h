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
#include "types.h"

namespace tinytorch
{


struct StorageImpl
{
    StorageImpl(int64_t size, Device device);
    StorageImpl(void* data_ptr, int64_t size, Device device);

    StorageImpl& operator=(StorageImpl&& other) = default;
    StorageImpl& operator=(const StorageImpl&)  = delete;
    StorageImpl()                               = delete;
    StorageImpl(StorageImpl&& other)            = default;
    StorageImpl(const StorageImpl&)             = delete;
    ~StorageImpl();

    uint8_t* byte_ptr() { return (uint8_t*)data_ptr_; }

   protected:
    Device device_;
    bool has_ownership = false;
    void* data_ptr_ = nullptr;
    int64_t size_;
};



}  // namespace tinytorch
