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