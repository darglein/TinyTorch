/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

#include "torch/core/tensor_options.h"
#include "torch/tiny_torch_config.h"
namespace tinytorch
{
namespace cpu_impl
{

void conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups,
              Tensor result);
}  // namespace cpu_impl
}  // namespace tinytorch