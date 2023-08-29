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
// basic operators
void add_impl(Tensor a, Tensor b, Tensor result);
void add_impl(Tensor a, double b, Tensor result);
void sub_impl(Tensor a, Tensor b, Tensor result);
void sub_impl(Tensor a, double b, Tensor result);
void mult_impl(Tensor a, Tensor b, Tensor result);
void mult_impl(Tensor a, double b, Tensor result);
void div_impl(Tensor a, Tensor b, Tensor result);
void div_impl(double a, Tensor b, Tensor result);


// comparison operators (no grad needed)
void equal_impl(Tensor a, double b, Tensor result);
void less_impl(Tensor a, double b, Tensor result);
void greater_impl(Tensor a, double b, Tensor result);

// binary functions
void pow_impl(Tensor a, double b, Tensor result);
void pow_impl(Tensor a, Tensor b, Tensor result);


void min_impl(Tensor a, Tensor b, Tensor result);
void max_impl(Tensor a, Tensor b, Tensor result);
}
}  // namespace tinytorch