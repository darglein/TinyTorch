/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl_cuda_helper.h"
#include "tt_cuda.h"

namespace tinytorch
{
namespace cuda
{

static constexpr int MAX_DEVICES = 8;

static cudaStream_t& thread_local_stream(int device_id)
{
    static thread_local cudaStream_t strms[MAX_DEVICES] = {};
    return strms[device_id];
}

cudaStream_t getCurrentCUDAStream()
{
    return thread_local_stream(getDevice());
}


cudaStream_t getCUDAStream(Device device)
{
    CHECK_EQ(device.type(), kCUDA);
    return thread_local_stream(device.index());
}


void setCUDAStreamForThisThread(cudaStream_t stream)
{
    thread_local_stream(getDevice()) = stream;
}

int getDevice()
{
    int device_index;
    CHECK_CUDA_ERROR(cudaGetDevice(&device_index));
    CHECK_LT(device_index, MAX_DEVICES);
    return device_index;
}

void setDevice(int device_index)
{
    CHECK_CUDA_ERROR(cudaSetDevice(device_index));
}

cudaEvent_t getNextEvent()
{
    constexpr int MAX_EVENTS                                              = 128;
    static thread_local int current_event[MAX_DEVICES]                    = {};
    static thread_local cudaEvent_t event_buffer[MAX_DEVICES][MAX_EVENTS] = {};

    int device   = getDevice();
    int& current = current_event[device];
    current      = (current + 1) % MAX_EVENTS;

    cudaEvent_t& event = event_buffer[device][current];

    if (!event)
    {
        cudaEventCreate(&event);
    }

    return event;
}

}  // namespace cuda
}  // namespace tinytorch