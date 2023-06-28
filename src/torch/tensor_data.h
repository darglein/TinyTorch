/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "assert.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tiny_torch_config.h"

namespace tinytorch
{


enum ScalarType
{
    kFloat,
};

struct TensorData
{
    TensorData(std::vector<int64_t> sizes, ScalarType type);

    int64_t numel()
    {
        int64_t res = 1;
        for (auto v : _sizes) res *= v;
        return res;
    }

    void* data = nullptr;
    std::vector<int64_t> _sizes;
    std::vector<int64_t> _strides;
    ScalarType _type;
};



}  // namespace tinytorch
