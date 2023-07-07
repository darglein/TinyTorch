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


TINYTORCH_API void copy(Tensor src, Tensor target);

TINYTORCH_API void fill(Tensor& t, double value);
TINYTORCH_API void uniform(Tensor& t);
TINYTORCH_API void uniform_int(Tensor& t, int low, int high);

inline std::pair<Tensor,Tensor> sort(Tensor t, int64_t a)
{
    throw std::runtime_error("not implemented");
    return {};
}


// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor square(Tensor a);

inline Tensor log(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor log1p(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor exp(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
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

inline Tensor sum(Tensor a, SizeType s)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor norm(Tensor a, int64_t norm, int64_t dim, bool keep)
{
    throw std::runtime_error("not implemented");
    return {};
}

TINYTORCH_API Tensor to(Tensor b, ScalarType other_type);


TINYTORCH_API Tensor sum(Tensor a);

inline Tensor stack(const std::vector<Tensor>& a)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor cat(const std::vector<Tensor>& a, int64_t dim)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor index_select(Tensor input, int64_t dim, Tensor index)
{
    throw std::runtime_error("not implemented");
    return {};
}

inline void load(Tensor&, std::string)
{
    throw std::runtime_error("not implemented");
}


}  // namespace tinytorch