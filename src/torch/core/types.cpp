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
        "kCPU",
        "kCUDA",
    };
    strm << type_names[(int)type.type];
    return strm;
}

std::ostream& operator<<(std::ostream& strm, ScalarType type)
{
    std::vector<std::string> type_names = {
        "kUint8",
        "kInt16",
        "kInt32",
        "kInt64",
        "kFloat16",
        "kFloat32",
        "kFloat64",
    };
    strm << type_names[(int)type];
    return strm;
}


}  // namespace tinytorch
