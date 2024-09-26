/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl_cuda_helper.h"
#include "tt_cuda.h"

#include <numeric>

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
        CHECK_CUDA_ERROR(cudaEventCreate(&event));
    }

    return event;
}


std::vector<Device> GetCudaDevicesFromDeviceList(std::vector<int> device_list)
{
    int cuda_device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&cuda_device_count));

    if (cuda_device_count == 0)
    {
        throw std::runtime_error("No CUDA capable device found\n");
    }

    std::sort(device_list.begin(), device_list.end());
    device_list.erase(std::unique(device_list.begin(), device_list.end()), device_list.end());

    if (device_list.empty())
    {
        std::cout << "Parameter 'device_list' is empty. Defaulting to device 0\n";
        device_list.push_back(0);
    }
    if (device_list[0] == -1)
    {
        device_list.resize(cuda_device_count);
        std::iota(device_list.begin(), device_list.end(), 0);
    }

    std::vector<Device> result;
    for (int index : device_list)
    {
        if (index < 0)
        {
            throw std::runtime_error("Invalid negative device_id " + std::to_string(index) +
                                   ". Only allowed negative number is -1, in which case all GPUs are used.");
        }

        if (index < cuda_device_count)
        {
            result.push_back(Device(kCUDA, index));
        }
        else
        {
            std::cout << "Ignoring device id " << index << ". Only " << cuda_device_count << " GPUs are available.\n";
        }
    }

    if (result.empty())
    {
        throw std::runtime_error("No device id in 'device_list' matches a valid GPU id.");
    }



    return result;
}


}  // namespace cuda
}  // namespace tinytorch