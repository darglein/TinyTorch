/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "tensor.h"

namespace tinytorch
{
namespace nn
{

struct Module
{
    std::map<std::string, Tensor> named_parameters(){

        throw std::runtime_error("not implemented");
        return {};
    }
};

struct AnyModule
{

};
}  // namespace nn
}  // namespace tinytorch
