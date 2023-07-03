/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

#ifdef TT_HAS_CUDA
#    include <cuda_runtime.h>
#endif


namespace tinytorch
{
StorageImpl::StorageImpl(int64_t size, Device device) : size_(size), device_(device)
{
    if (device_ == kCPU)
    {
        data_ptr_ = malloc(size);
    }
    else
    {
#ifdef TT_HAS_CUDA
        cudaMalloc(&data_ptr_, size);
#else
        assert(false);
#endif
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
#ifdef TT_HAS_CUDA
        cudaFree(data_ptr_);
#else
        assert(false);
#endif
    }
}

}  // namespace tinytorch