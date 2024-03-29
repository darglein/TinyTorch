/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

namespace tinytorch
{


TINYTORCH_API Tensor pow(Tensor a, double b);
TINYTORCH_API Tensor pow(Tensor a, Tensor b);

TINYTORCH_API Tensor prod(Tensor a, int64_t dim, bool keepdim = false);
TINYTORCH_API Tensor cumprod(Tensor a, int64_t dim);
TINYTORCH_API Tensor cumsum(Tensor a, int64_t dim);

TINYTORCH_API Tensor min(Tensor a, Tensor b);
TINYTORCH_API Tensor max(Tensor a, Tensor b);
TINYTORCH_API Tensor min(Tensor a);
TINYTORCH_API Tensor max(Tensor a);
TINYTORCH_API std::pair<Tensor, Tensor> min(Tensor a, int64_t dim, bool keepdim = false);
TINYTORCH_API std::pair<Tensor, Tensor> max(Tensor a, int64_t dim, bool keepdim = false);


TINYTORCH_API Tensor clamp(Tensor a, double low, double high);
TINYTORCH_API void clamp_(Tensor& a, double low, double high);

TINYTORCH_API Tensor norm(Tensor a, int64_t norm, int64_t dim, bool keep);



TINYTORCH_API Tensor std(Tensor a);
TINYTORCH_API Tensor std(Tensor a, int64_t dim);

TINYTORCH_API Tensor abs_sum(Tensor a);

TINYTORCH_API Tensor sum(Tensor a);
TINYTORCH_API Tensor sum(Tensor a, int64_t dim, bool keepdim);
TINYTORCH_API Tensor sum(Tensor a, SizeType s, bool keepdim = false);

TINYTORCH_API Tensor mean(Tensor a);
TINYTORCH_API Tensor mean(Tensor a, int64_t dim, bool keepdim);
TINYTORCH_API Tensor mean(Tensor a, SizeType s, bool keepdim = false);


}  // namespace tinytorch