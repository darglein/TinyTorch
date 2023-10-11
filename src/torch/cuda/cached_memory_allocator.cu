/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <mutex>

#include "cached_memory_allocator.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>


namespace tinytorch
{
namespace cuda
{

std::mutex mu;


static void* malloc_async(int64_t size)
{
    void* ptr;
    auto strm       = cuda::getCurrentCUDAStream();
    auto cuda_error = cudaMallocAsync(&ptr, size, strm);
    if (cuda_error == cudaErrorMemoryAllocation)
    {
        size_t mem_free, mem_total;
        cudaMemGetInfo(&mem_free, &mem_total);
        CHECK_NE(cuda_error, cudaErrorMemoryAllocation)
            << " CUDA out of memory!\n"
            << "     Tried to allocate " << (size / 1000.0 / 1000.0) << "MB\n"
            << "     Free memory " << (mem_free / 1000.0 / 1000.0) << "MB\n"
            << "     Total memory " << (mem_total / 1000.0 / 1000.0) << "MB\n";
    }
    CHECK_CUDA_ERROR(cuda_error);
    CHECK_NOTNULL(ptr);
    return ptr;
}

void* cuda_cached_malloc(int64_t size)
{
    if (size == 0)
    {
        return nullptr;
    }
    std::unique_lock l(mu);

    auto ptr = malloc_async(size);

//    if ((size / 1000.0 / 1000.0) > 100)
//    {
//        std::cout << "Allocate CUDA Memory: " << (size / 1000.0 / 1000.0) << "MB\n";
//    }

    return ptr;
}
void cuda_cached_free(void* ptr)
{
    std::unique_lock l(mu);
    cudaFreeAsync(ptr, cuda::getCurrentCUDAStream());
}
void CUDACachingAllocator::emptyCache()
{
    // this frees unused values for the async allocator
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

}  // namespace cuda

}  // namespace tinytorch