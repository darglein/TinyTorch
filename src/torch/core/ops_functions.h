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
TINYTORCH_API Tensor repeat_interleave(Tensor t, int64_t count);
TINYTORCH_API Tensor transpose(Tensor t, int64_t dim0, int64_t dim1);

TINYTORCH_API Tensor permute(Tensor t, const SizeType& size);

TINYTORCH_API void copy(Tensor src, Tensor target);


TINYTORCH_API void fill(Tensor& t, double value);
TINYTORCH_API void fill(Tensor& t, Tensor value);
TINYTORCH_API void fill(Tensor& t, Tensor values, int dim);


TINYTORCH_API void uniform(Tensor& t, double mi = 0, double ma = 1);
TINYTORCH_API void uniform_int(Tensor& t, int low, int high);

inline std::pair<Tensor, Tensor> sort(Tensor t, int64_t a)
{
    throw std::runtime_error("not implemented");
    return {};
}

TINYTORCH_API Tensor clone(Tensor a);

TINYTORCH_API Tensor to(Tensor a, ScalarType other_type);
TINYTORCH_API Tensor to(Tensor a, Device other_type);



TINYTORCH_API Tensor slice(Tensor a, int64_t dim, int64_t start, int64_t end, int64_t step);
TINYTORCH_API Tensor stack(const std::vector<Tensor>& a);
TINYTORCH_API Tensor cat(const std::vector<Tensor>& a, int64_t dim);
TINYTORCH_API Tensor index_select(Tensor input, int64_t dim, Tensor index);
TINYTORCH_API Tensor index_add(Tensor input, int64_t dim, Tensor index, Tensor data);

inline void load(Tensor&, std::string)
{
    throw std::runtime_error("not implemented");
}


}  // namespace tinytorch