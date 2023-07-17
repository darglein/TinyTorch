/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

#include "tensor_options.h"
#include "torch/tiny_torch_config.h"
namespace tinytorch
{
TINYTORCH_API std::ostream& operator<<(std::ostream& strm, Tensor t);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor operator+(Tensor a, Tensor b);
TINYTORCH_API Tensor operator+(Tensor a, double b);
TINYTORCH_API Tensor operator+(double a, Tensor b);

TINYTORCH_API Tensor operator-(Tensor b);
TINYTORCH_API Tensor operator-(Tensor a, Tensor b);
TINYTORCH_API Tensor operator-(Tensor a, double b);
TINYTORCH_API Tensor operator-(double a, Tensor b);

TINYTORCH_API Tensor operator*(Tensor a, Tensor b);
TINYTORCH_API Tensor operator*(double a, Tensor b);
TINYTORCH_API Tensor operator*(Tensor a, double b);

TINYTORCH_API Tensor operator/(Tensor a, Tensor b);
TINYTORCH_API Tensor operator/(Tensor a, double b);
TINYTORCH_API Tensor operator/(double a, Tensor b);

TINYTORCH_API Tensor operator==(Tensor a, double b);
TINYTORCH_API Tensor operator<(Tensor a, double b);
TINYTORCH_API Tensor operator>(Tensor a, double b);

TINYTORCH_API Tensor operator+=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator+=(Tensor a, double b);
TINYTORCH_API Tensor operator-=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator-=(Tensor a, double b);
TINYTORCH_API Tensor operator*=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator*=(Tensor a, double b);
TINYTORCH_API Tensor operator/=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator/=(Tensor a, double b);



}  // namespace tinytorch