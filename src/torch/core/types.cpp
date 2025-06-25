/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "types.h"
#ifdef TT_HAS_CUDA
#    include "torch/cuda/tt_cuda.h"
#endif

namespace tinytorch
{
std::ostream& operator<<(std::ostream& strm, Device type)
{
    std::vector<std::string> type_names = {
        "cpu",
        "cuda",
        "undefined",
    };
    strm << type_names[(int)type.type()];
    if (type.type() == kCUDA)
    {
        strm << ":" << type.index();
    }
    return strm;
}

std::ostream& operator<<(std::ostream& strm, ScalarType type)
{
    std::vector<std::string> type_names = {
        "kUint8", "kInt16", "kInt32", "kInt64", "kFloat16", "float", "double", "kUInt16", "kUnknown",
    };
    strm << type_names[(int)type];
    return strm;
}


Device::Device(DeviceType type, int index) : _type(type), _index(index) 
{
    if (type == kCPU)
    {
        _index = 0;
    }
    else if (_index < 0)
    {
#ifdef TT_HAS_CUDA
        _index = cuda::getDevice();
#endif
    }
}

}  // namespace tinytorch
