/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */

#pragma once

#include "tensor.h"
namespace tinytorch
{


Tensor square(Tensor a);

Tensor operator-(Tensor a, Tensor b);
Tensor operator+(Tensor a, Tensor b);
Tensor operator*(Tensor a, Tensor b);

Tensor sum(Tensor a);


}  // namespace tinytorch