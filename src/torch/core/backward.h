#pragma once
#include "torch/core/tensor.h"

namespace tinytorch
{


TINYTORCH_API void backward(Tensor loss, Tensor grad = {}, bool retain_grad = false);

}  // namespace tinytorch