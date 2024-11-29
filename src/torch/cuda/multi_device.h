#pragma once

#include "torch/cuda/tt_cuda.h"

namespace tinytorch
{
namespace cuda
{

TINYTORCH_API bool HasP2PCopy();

TINYTORCH_API bool IsCudaPeerToPeerAvailable(Device device0, Device device1);

// Must be called for all active devices to enable torch-p2p copies
TINYTORCH_API bool EnableCudaPeerToPeer(const std::vector<Device>& devices);
TINYTORCH_API void DisableCudaPeerToPeer(const std::vector<Device>& devices);



struct MultiDeviceTensor
{
    std::vector<Tensor> data;

    // Init with undefined tensor
    MultiDeviceTensor() { data.push_back({}); }
    MultiDeviceTensor(Tensor d) { data.push_back(d); }

    Tensor* operator->() { return &data.front(); }

    // implicit cast to main_device tensor, to make use of existing functions
    operator Tensor&() { return data.front(); }
    operator const Tensor&() const { return data.front(); }
};


}  // namespace cuda
}  // namespace tinytorch