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

#include "tensor_data.h"
#include "tiny_torch_config.h"

namespace tinytorch
{



struct TINYTORCH_API TensorOptions
{
    Device device_    = kCPU;
    ScalarType dtype_ = kFloat;

    // Bitmask required here to get this to fit inside 32 bits (or even 64 bits,
    // for that matter)
    bool requires_grad_ : 1;
    bool pinned_memory_ : 1;

    bool has_device_ : 1;
    bool has_dtype_ : 1;
    bool has_layout_ : 1;
    bool has_requires_grad_ : 1;
    bool has_pinned_memory_ : 1;
    bool has_memory_format_ : 1;

    TensorOptions() {}
    TensorOptions(Device device) : device_(device) {}
    TensorOptions(ScalarType dtype) : dtype_(dtype) {}


    TensorOptions& device(Device d)
    {
        device_ = d;
        return *this;
    }
    TensorOptions& dtype(ScalarType d)
    {
        dtype_ = d;
        return *this;
    }
};


}  // namespace tinytorch
