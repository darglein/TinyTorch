/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "tensor.h"

#include "tensor_options.h"
#include "tiny_torch_config.h"
namespace tinytorch
{
// Basic tensor generation
TINYTORCH_API Tensor full(const std::vector<int64_t>& sizes, float value, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor ones(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor empty(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor zeros(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor rand(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());

TINYTORCH_API Tensor full_like(Tensor t, float value);
TINYTORCH_API Tensor ones_like(Tensor t);
TINYTORCH_API Tensor empty_like(Tensor t);
TINYTORCH_API Tensor zeros_like(Tensor t);
TINYTORCH_API Tensor rand_like(Tensor t);


inline Tensor from_blob(void* data, std::vector<int64_t> sizes,std::vector<int64_t> stride, ScalarType type)
{

    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API std::ostream& operator<<(std::ostream& strm, Tensor t);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor square(Tensor a);
TINYTORCH_API Tensor operator-(Tensor a, Tensor b);
TINYTORCH_API Tensor operator+(Tensor a, Tensor b);
TINYTORCH_API Tensor operator*(Tensor a, Tensor b);
inline Tensor operator/(Tensor a, Tensor b){

    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator*(double a, Tensor b){

    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator+(double a, Tensor b){

    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API Tensor sum(Tensor a);

TINYTORCH_API Tensor& operator+=(Tensor& a, Tensor b);


inline Tensor cat(std::vector<Tensor> a, int64_t dim)
{
    throw std::runtime_error("not implemented");
    return {};
}

// Internal implementation of forward/backward
// Should NOT be called by the user
void fill_impl(Tensor a, float value);
Tensor square_impl(Tensor a);
Tensor sub_impl(Tensor a, Tensor b);
Tensor add_impl(Tensor a, Tensor b);
Tensor mult_impl(Tensor a, Tensor b);
Tensor sum_impl(Tensor a);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output);
std::vector<Tensor> add_backward_impl(Tensor grad_output);
std::vector<Tensor> sub_backward_impl(Tensor grad_output);
std::vector<Tensor> sum_backward_impl(const std::vector<int64_t>& input_sizes, Tensor grad_output);

}  // namespace tinytorch