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

TINYTORCH_API Tensor repeat(Tensor t, SizeType sizes);

TINYTORCH_API void copy(Tensor src, Tensor target);

TINYTORCH_API void fill(Tensor& t, double value);
TINYTORCH_API void uniform(Tensor& t);
TINYTORCH_API void uniform_int(Tensor& t, int low, int high);

inline std::pair<Tensor,Tensor> sort(Tensor t, int64_t a)
{
    throw std::runtime_error("not implemented");
    return {};
}


TINYTORCH_API Tensor to(Tensor b, ScalarType other_type);



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
TINYTORCH_API Tensor index_select(Tensor input, int64_t dim, Tensor index);

inline void load(Tensor&, std::string)
{
    throw std::runtime_error("not implemented");
}


}  // namespace tinytorch