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

// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor square(Tensor a);
TINYTORCH_API Tensor sqrt(Tensor a);

TINYTORCH_API Tensor log(Tensor a);
TINYTORCH_API Tensor log1p(Tensor a);
TINYTORCH_API Tensor exp(Tensor a);

TINYTORCH_API Tensor round(Tensor b);

inline Tensor sign(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor pow(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor pow(Tensor a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor sin(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor cos(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor relu(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor sigmoid(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor softplus(Tensor b, double beta)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor prod(Tensor b, int64_t dim)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor cumprod(Tensor b, int64_t dim)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor cumsum(Tensor b, int64_t dim)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API Tensor min(Tensor a, Tensor b);
TINYTORCH_API Tensor max(Tensor a, Tensor b);
TINYTORCH_API Tensor min(Tensor a);
TINYTORCH_API Tensor max(Tensor a);
TINYTORCH_API std::pair<Tensor, Tensor> min(Tensor a, int64_t dim, bool keepdim = false);
TINYTORCH_API std::pair<Tensor, Tensor> max(Tensor a, int64_t dim, bool keepdim = false);


TINYTORCH_API Tensor clamp(Tensor a, double low, double high);
TINYTORCH_API void clamp_(Tensor& a, double low, double high);

TINYTORCH_API Tensor norm(Tensor a, int64_t norm, int64_t dim, bool keep);
TINYTORCH_API Tensor abs(Tensor a);


TINYTORCH_API Tensor std(Tensor a);
TINYTORCH_API Tensor std(Tensor a, int64_t dim);

TINYTORCH_API Tensor sum(Tensor a);
TINYTORCH_API Tensor sum(Tensor a, int64_t dim, bool squeeze_dim);
TINYTORCH_API Tensor sum(Tensor a, SizeType s);

TINYTORCH_API Tensor mean(Tensor a);
TINYTORCH_API Tensor mean(Tensor a, int64_t dim, bool squeeze_dim);
TINYTORCH_API Tensor mean(Tensor a, SizeType s);


}  // namespace tinytorch