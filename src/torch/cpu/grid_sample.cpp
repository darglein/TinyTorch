/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "grid_sample.h"

namespace tinytorch
{
namespace cpu_impl
{

void grid_sample_2d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{
    CHECK(false);
}
void grid_sample_2d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{
    CHECK(false);
}
void grid_sample_3d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{    CHECK(false);
}
void grid_sample_3d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{    CHECK(false);
}
}  // namespace cpu_impl

}  // namespace tinytorch
