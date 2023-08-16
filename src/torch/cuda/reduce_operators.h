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

// full reductions
void sum_impl(Tensor a, Tensor& result);
void min_impl(Tensor a, Tensor& result);
void max_impl(Tensor a, Tensor& result);

// 1D reductions
void min_impl(Tensor a, int64_t dim, Tensor& result, Tensor& indices);
void sum_impl(Tensor a, int64_t dim, Tensor& result);
void max_impl(Tensor a, int64_t dim, Tensor& result, Tensor& indices);
void prod_impl(Tensor a, int64_t dim, Tensor& result);

// Scans
void cumprod_impl(Tensor a, int64_t dim, Tensor& result);
void cumsum_impl(Tensor a, int64_t dim, Tensor& result);
}  // namespace cuda_impl
}  // namespace tinytorch