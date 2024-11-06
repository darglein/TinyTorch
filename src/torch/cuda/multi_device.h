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


}
}