/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

namespace tinytorch
{

// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor abs(Tensor a);
TINYTORCH_API Tensor round(Tensor b);
TINYTORCH_API Tensor sqrt(Tensor a);
TINYTORCH_API Tensor square(Tensor a);
TINYTORCH_API Tensor log(Tensor a);
TINYTORCH_API Tensor log1p(Tensor a);
TINYTORCH_API Tensor exp(Tensor a);

TINYTORCH_API Tensor sign(Tensor b);
TINYTORCH_API Tensor sin(Tensor b);
TINYTORCH_API Tensor cos(Tensor b);

TINYTORCH_API Tensor relu(Tensor b);
TINYTORCH_API Tensor sigmoid(Tensor b);
TINYTORCH_API Tensor softplus(Tensor b, double beta);



}  // namespace tinytorch