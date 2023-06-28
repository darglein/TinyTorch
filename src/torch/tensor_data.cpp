/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor_data.h"

tinytorch::TensorData::TensorData(std::vector<int64_t> sizes, ScalarType type) : _sizes(sizes), _type(type)
{
    auto size_per_element = 4;
    data = malloc(size_per_element * numel());
}
