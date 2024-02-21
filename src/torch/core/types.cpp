/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "types.h"

namespace tinytorch
{
std::ostream& operator<<(std::ostream& strm, Device type)
{
    std::vector<std::string> type_names = {
        "cpu",
        "cuda",
    };
    strm << type_names[(int)type.type()];
    if (type.type() == kCUDA)
    {
        strm << ":" << type.device_index;
    }
    return strm;
}

std::ostream& operator<<(std::ostream& strm, ScalarType type)
{
    std::vector<std::string> type_names = {
        "kUint8", "kInt16", "kInt32", "kInt64", "kFloat16", "float", "double", "kUInt16",
    };
    strm << type_names[(int)type];
    return strm;
}


}  // namespace tinytorch
