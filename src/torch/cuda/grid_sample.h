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
namespace cuda_impl
{

void grid_sample_2d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor& result);
void grid_sample_2d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result);

void grid_sample_3d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor& result);
void grid_sample_3d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result);

}  // namespace cpu_impl
}  // namespace tinytorch