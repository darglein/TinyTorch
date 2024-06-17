/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

#include "torch/cuda/cached_memory_allocator.h"

#ifdef TT_HAS_CUDA
#    include "torch/cuda/tt_cuda.h"
#    include "torch/cuda/ops_impl_cuda_helper.h"
#    include <cuda_runtime.h>
#endif


namespace tinytorch
{
StorageImpl::StorageImpl(int64_t size, TensorOptions options) : size_(size), options_(options)
{
    if (options_.device_ == kCPU)
    {
#ifdef TT_HAS_CUDA
        if (options.pinned_memory_)
        {
            cudaError_t cuda_error = cudaMallocHost(&data_ptr_, size);
            if (cuda_error == cudaErrorMemoryAllocation)
            {
                size_t mem_free, mem_total;
                cudaMemGetInfo(&mem_free, &mem_total);
                std::cerr << " CUDA out of memory!\n"
                          << "     Tried to allocate " << (size / 1000.0 / 1000.0) << "MB of pinned memory\n";

                throw std::runtime_error(std::string("CUDA pinned memory allocation error: ") +
                                         cudaGetErrorString(cuda_error));
            }
        }
        else
        {
            data_ptr_ = malloc(size);
        }
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
        cuda::DeviceGuard g(options_.device_);

        data_ptr_ = cuda::cuda_cached_malloc(size);
#    if TT_DEBUG
        CHECK_CUDA_ERROR(cudaMemset(data_ptr_, 0xabababab, size));
#    endif
        has_ownership = true;
#else
        CHECK(false);
#endif
    }
}


StorageImpl::StorageImpl(void* data_ptr, int64_t size, TensorOptions options) : size_(size), options_(options)
{
    data_ptr_     = data_ptr;
    has_ownership = false;
}


StorageImpl::~StorageImpl()
{
    if (has_ownership == true)
    {
        if (options_.device_ == kCPU)
        {
#ifdef TT_HAS_CUDA
            if (options_.pinned_memory_)
            {
                cudaFreeHost(data_ptr_);
            }
            else
            {
                free(data_ptr_);
            }
#else
            free(data_ptr_);
#endif
        }
        else
        {
#ifdef TT_HAS_CUDA
            cuda::DeviceGuard g(options_.device_);

            cuda::cuda_cached_free(data_ptr_);
#else
            CHECK(false);
#endif
        }
    }
}

}  // namespace tinytorch