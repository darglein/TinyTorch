/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

namespace tinytorch
{
TINYTORCH_API std::ostream& operator<<(std::ostream& strm, Tensor t);

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