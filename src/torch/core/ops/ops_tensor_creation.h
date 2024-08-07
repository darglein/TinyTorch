/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

namespace tinytorch
{
TINYTORCH_API void manual_seed(int64_t seed);
TINYTORCH_API std::mt19937& generator();


TINYTORCH_API Tensor empty(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor empty_like(Tensor t, TensorOptions options);
TINYTORCH_API Tensor empty_like(Tensor t);

TINYTORCH_API Tensor full(const SizeType& sizes, float value, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor full_like(Tensor t, float value, TensorOptions options);
TINYTORCH_API Tensor full_like(Tensor t, float value);

TINYTORCH_API Tensor zeros(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor zeros_like(Tensor t, TensorOptions options);
TINYTORCH_API Tensor zeros_like(Tensor t);

TINYTORCH_API Tensor ones(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor ones_like(Tensor t, TensorOptions options);
TINYTORCH_API Tensor ones_like(Tensor t);

TINYTORCH_API Tensor rand(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor randint(int low, int high, const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor rand_like(Tensor t);

// end is inclusive
TINYTORCH_API Tensor range(double start, double end, double step = 1,  TensorOptions options = TensorOptions());

TINYTORCH_API Tensor from_blob(void* data, const SizeType& sizes, const SizeType& stride, ScalarType type = kFloat);
TINYTORCH_API Tensor from_blob(void* data, const SizeType& sizes, ScalarType type = kFloat);
TINYTORCH_API Tensor from_blob(void* data, const SizeType& sizes, TensorOptions options);
TINYTORCH_API Tensor from_blob(void* data, const SizeType& sizes, const SizeType& stride, TensorOptions options);


}  // namespace tinytorch