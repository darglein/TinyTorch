/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

#ifdef TT_HAS_CUDA
#    include <cuda_runtime.h>
#include "torch/cuda/ops_impl_cuda_helper.h"
#endif


namespace tinytorch
{
StorageImpl::StorageImpl(int64_t size, Device device) : size_(size), device_(device)
{
    if (device_ == kCPU)
    {
        data_ptr_     = malloc(size);
        has_ownership = true;
    }
    else
    {
#ifdef TT_HAS_CUDA
        CHECK_CUDA_ERROR(cudaMalloc(&data_ptr_, size));
        has_ownership = true;
#else
        CHECK(false);
#endif
    }
}


StorageImpl::StorageImpl(void* data_ptr, int64_t size, Device device) : size_(size), device_(device)
{
    data_ptr_     = data_ptr;
    has_ownership = false;
}


StorageImpl::~StorageImpl()
{
    if (has_ownership == true)
    {
        if (device_ == kCPU)
        {
            free(data_ptr_);
        }
        else
        {
#ifdef TT_HAS_CUDA
            cudaFree(data_ptr_);
            // CHECK_CUDA_ERROR(cudaFree(data_ptr_));
#else
            CHECK(false);
#endif
        }
    }
}

}  // namespace tinytorch