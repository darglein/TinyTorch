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
TINYTORCH_API Tensor square(Tensor a);
TINYTORCH_API Tensor operator+(Tensor a, Tensor b);

TINYTORCH_API Tensor operator+(Tensor a, double b);
inline Tensor operator+(double a, Tensor b)
{
   return b + a;
}

TINYTORCH_API Tensor operator-(Tensor a, Tensor b);
inline Tensor operator-(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator-(double a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API Tensor operator*(Tensor a, Tensor b);
TINYTORCH_API Tensor operator*(double a, Tensor b);
inline Tensor operator*(Tensor a, double b)
{
    return b * a;
}
inline Tensor operator/(Tensor a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator/(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator/(double a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API Tensor operator-(Tensor b);
TINYTORCH_API Tensor operator==(Tensor a, double b);
inline Tensor operator<(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator>(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}

TINYTORCH_API Tensor operator+=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator+=(Tensor a, double b);
TINYTORCH_API Tensor operator-=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator-=(Tensor a, double b);
TINYTORCH_API Tensor operator*=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator*=(Tensor a, double b);
TINYTORCH_API Tensor operator/=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator/=(Tensor a, double b);



}  // namespace tinytorch