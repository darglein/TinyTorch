/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

#include "torch/cuda/cached_memory_allocator.h"

#ifdef TT_HAS_CUDA
#    include "torch/cuda/ops_impl_cuda_helper.h"
#    include <cuda_runtime.h>
#endif


namespace tinytorch
{
StorageImpl::StorageImpl(int64_t size, Device device) : size_(size), device_(device)
{
    if (device_ == kCPU)
    {
#ifdef TT_HAS_CUDA3465
        cudaMallocHost(&data_ptr_, size);
#else
        data_ptr_ = malloc(size);
#endif
#if TT_DEBUG
        memset(data_ptr_, 0xabababab, size);
#endif
        has_ownership = true;
    }
    else
    {
#ifdef TT_HAS_CUDA
        data_ptr_ = cuda::cuda_cached_malloc(size);
#    if TT_DEBUG
        cudaMemset(data_ptr_, 0xabababab, size);
#    endif
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
#ifdef TT_HAS_CUDA456
            cudaFreeHost(data_ptr_);
#else
            free(data_ptr_);
#endif
        }
        else
        {
#ifdef TT_HAS_CUDA
            cuda::cuda_cached_free(data_ptr_);
#else
            CHECK(false);
#endif
        }
    }
}

}  // namespace tinytorch