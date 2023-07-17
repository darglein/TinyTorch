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

void print_impl_cpu(std::ostream& strm, Tensor t);
// basic operators
void add_impl_cpu(Tensor a, Tensor b, Tensor& result);
void add_impl_cpu(Tensor a, double b, Tensor& result);
void sub_impl_cpu(Tensor a, Tensor b, Tensor& result);
void mult_impl_cpu(Tensor a, Tensor b, Tensor& result);
void mult_impl_cpu(Tensor a, double b, Tensor& result);
void div_impl_cpu(Tensor a, Tensor b, Tensor& result);
void div_impl_cpu(double a, Tensor b, Tensor& result);

// comparison operators (no grad needed)
void equal_impl_cpu(Tensor a, double b, Tensor& result);
void less_impl_cpu(Tensor a, double b, Tensor& result);
void greater_impl_cpu(Tensor a, double b, Tensor& result);

}  // namespace tinytorch