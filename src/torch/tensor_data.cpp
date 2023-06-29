/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

namespace TINY_TORCH_NAMESPACE
{
StorageImpl::StorageImpl(int64_t size, Device device) : size_(size), device_(device)
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
StorageImpl::~StorageImpl()
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

}