/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "assert.h"
#include "types.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tensor_options.h"
#include "torch/tiny_torch_config.h"

namespace tinytorch
{


struct StorageImpl
{
    StorageImpl(int64_t size, TensorOptions options);
    StorageImpl(void* data_ptr, int64_t size, uint64_t alloc_info, TensorOptions options);

    StorageImpl& operator=(StorageImpl&& other) = default;
    StorageImpl& operator=(const StorageImpl&)  = delete;
    StorageImpl()                               = delete;
    StorageImpl(StorageImpl&& other)            = default;
    StorageImpl(const StorageImpl&)             = delete;
    ~StorageImpl();

    uint8_t* byte_ptr() { return (uint8_t*)data_ptr_; }

   uint64_t allocinfo() const { return alloc_info; }
   protected:
    TensorOptions options_;
    bool has_ownership   = false;
    void* data_ptr_      = nullptr;
    uint64_t alloc_info = 0;
    int64_t size_;
};



}  // namespace tinytorch
