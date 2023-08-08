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

void grid_sample_2d_impl(Tensor data, Tensor uv, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor& result);

}  // namespace cpu_impl
}  // namespace tinytorch