/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/tensor.h"

#include "cached_memory_allocator.h"
#include "cuda_runtime.h"

#ifdef __CUDACC__
// Warning: This assert is pretty slow, so make sure it is only active during debug

#    undef assert

#    if TT_DEBUG
inline __device__ void __assert_fail_cuda(const char* __assertion, const char* __file, unsigned int __line,
                                          const char* __function)
{
    printf("cuda assert failed in %s:%d func: %s\n block:     %d %d %d\n thread:    %d %d %d\n Assertion: %s\n", __file,
           __line, __function, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, __assertion);
    __trap();
}

#        define CUDA_KERNEL_ASSERT(cond)                                                            \
            if ((!(cond)))                                                                          \
            {                                                                                       \
                __assert_fail_cuda(#cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
            }
#    else
#        define CUDA_KERNEL_ASSERT(cond) (void)0
#    endif

#else
#    define CUDA_KERNEL_ASSERT(cond) assert(cond)
#endif



#if defined(__CUDA_RUNTIME_H__) || defined(__CUDACC__)
#endif

namespace tinytorch
{
namespace cuda
{
TINYTORCH_API int& getTotalNumEventsUsed();
TINYTORCH_API cudaEvent_t getNextEvent();

TINYTORCH_API cudaStream_t getCurrentCUDAStream();
TINYTORCH_API cudaStream_t getCUDAStream(Device device);


TINYTORCH_API void setCUDAStreamForThisThread(cudaStream_t stream);

TINYTORCH_API int getDevice();
TINYTORCH_API void setDevice(int device_index);

struct TINYTORCH_API DeviceGuard
{
    DeviceGuard() = delete;
    DeviceGuard(Device device)
    {
        CHECK_EQ(device.type(), kCUDA);

        original_device_index_ = getDevice();
        if (device.index() >= 0 && device.index() != original_device_index_)
        {
            setDevice(device._index);
        }
    }
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&&)      = delete;
    ~DeviceGuard() { setDevice(original_device_index_); }

    int original_device_index_;
};



TINYTORCH_API std::vector<Device> GetCudaDevicesFromDeviceList(std::vector<int> device_list);

TINYTORCH_API void ReportCudaError(cudaError_t cudaErrorCode, std::string function);
}  // namespace cuda
}  // namespace tinytorch

#define TT_CHECK_CUDA_ERROR(cudaFunction)                                   \
    {                                                                       \
        cudaError_t cudaErrorCode = cudaFunction;                           \
        if (cudaErrorCode != cudaSuccess)                                   \
        {                                                                   \
            tinytorch::cuda::ReportCudaError(cudaErrorCode, #cudaFunction); \
        }                                                                   \
    }
