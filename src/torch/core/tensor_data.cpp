/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

#include "torch/cuda/cached_memory_allocator.h"

#ifdef TT_HAS_CUDA
// #    include <sys/mman.h>

#    include "torch/cuda/ops_impl_cuda_helper.h"
#    include "torch/cuda/tt_cuda.h"
#    include <cuda_runtime.h>
#endif


static void* malloc_impl_host(int64_t size)
{
    auto ptr = malloc(size);
    // auto ptr = calloc(size, 1);
    //    auto ptr = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    // memset(ptr,0,size);
    return ptr;
}

static void free_impl_host(void* ptr, int64_t size)
{
    free(ptr);
    //    munmap(ptr, size);
}


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
            data_ptr_ = malloc_impl_host(size);
        }
#else
        data_ptr_ = malloc_impl_host(size);
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

        std::tie(data_ptr_, alloc_info) = cuda::cuda_cached_malloc(size);
#    if TT_DEBUG
        CHECK_CUDA_ERROR(cudaMemsetAsync(data_ptr_, 0xabababab, size, cuda::getCurrentCUDAStream()));
#    endif

        has_ownership = true;
#else
        CHECK(false);
#endif
    }
}


StorageImpl::StorageImpl(void* data_ptr, int64_t size, uint64_t alloc_info, TensorOptions options)
    : size_(size), alloc_info(alloc_info), options_(options)
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
                free_impl_host(data_ptr_, size_);
            }
#else
            free_impl_host(data_ptr_, size_);
#endif
        }
        else
        {
#ifdef TT_HAS_CUDA
            cuda::DeviceGuard g(options_.device_);

            cuda::cuda_cached_free(data_ptr_, alloc_info);
#else
            CHECK(false);
#endif
        }
    }
}

}  // namespace tinytorch