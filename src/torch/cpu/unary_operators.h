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
void abs_impl(Tensor a, Tensor result);
void round_impl(Tensor a, Tensor result);
void sqrt_impl(Tensor a, Tensor result);
void sqrt_backward_impl(Tensor a, Tensor grad_a, Tensor grad_result);
void log_impl(Tensor a, Tensor result);
void exp_impl(Tensor a, Tensor result);
void sign_impl(Tensor a, Tensor result);
void sin_impl(Tensor a, Tensor result);
void cos_impl(Tensor a, Tensor result);
void relu_impl(Tensor a, Tensor result);
void sigmoid_impl(Tensor a, Tensor result);
void sigmoid_backward_impl(Tensor a, Tensor grad_a, Tensor grad_result);
void softplus_impl(Tensor a, double beta, Tensor result);
void softplus_backward_impl(Tensor a, double beta, Tensor grad_a, Tensor grad_result);

}  // namespace cpu_impl
}  // namespace tinytorch