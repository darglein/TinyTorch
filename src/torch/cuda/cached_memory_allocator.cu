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

template <typename T, typename U>
constexpr T iAlignUp(T a, U b)
{
    static_assert(std::is_integral<T>::value && std::is_integral<U>::value, "only applicable to integral types");
    return (a % b != 0) ? (a - a % b + b) : a;
}

std::mutex mu;

void* cuda_cached_malloc(int64_t size)
{
    if (size == 0)
    {
        return nullptr;
    }
    std::unique_lock l(mu);
    void* ptr;
    auto cuda_error = cudaMallocAsync(&ptr, size, cuda::getCurrentCUDAStream());
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
void cuda_cached_free(void* ptr)
{
    std::unique_lock l(mu);
    cudaFreeAsync(ptr, cuda::getCurrentCUDAStream());
}
void CUDACachingAllocator::emptyCache() {}

}  // namespace cuda

}  // namespace tinytorch