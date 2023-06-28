/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"


tinytorch::StorageImpl::StorageImpl(int64_t size, tinytorch::DeviceType device) : size_(size), device_(device)
{
    if (device_ == kCPU)
    {
        data_ptr_ = malloc(size);
    }
    else
    {
        assert(false);
    }
}
tinytorch::StorageImpl::~StorageImpl()
{
    if (device_ == kCPU)
    {
        free(data_ptr_);
    }
    else
    {
        assert(false);
    }
}
