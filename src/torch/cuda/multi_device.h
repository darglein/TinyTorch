#pragma once

#include "torch/cuda/tt_cuda.h"

namespace tinytorch
{
namespace cuda
{


TINYTORCH_API bool IsCudaPeerToPeerAvailable(Device device0, Device device1);

TINYTORCH_API bool EnableCudaPeerToPeerVec(const std::vector<Device>& devices);
TINYTORCH_API void DisableCudaPeerToPeerVec(const std::vector<Device>& devices);

TINYTORCH_API bool EnableCudaPeerToPeer(Device device0, Device device1);
TINYTORCH_API void DisableCudaPeerToPeer(Device device0, Device device1);


}
}