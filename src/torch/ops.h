/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "tensor.h"

#include "tensor_options.h"
#include "tiny_torch_config.h"
namespace TINY_TORCH_NAMESPACE
{
// Basic tensor generation
TINYTORCH_API Tensor full(const std::vector<int64_t>& sizes, float value, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor ones(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor empty(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor zeros(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor rand(const std::vector<int64_t>& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor randint(int low, int high, const std::vector<int64_t>& sizes,
                             TensorOptions options = TensorOptions());

TINYTORCH_API Tensor full_like(Tensor t, float value);
TINYTORCH_API Tensor ones_like(Tensor t);
TINYTORCH_API Tensor empty_like(Tensor t);
TINYTORCH_API Tensor zeros_like(Tensor t);
TINYTORCH_API Tensor rand_like(Tensor t);


inline Tensor from_blob(void* data, const std::vector<int64_t>& sizes, const std::vector<int64_t>& stride,
                        ScalarType type = kFloat)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor from_blob(void* data, const std::vector<int64_t>& sizes, ScalarType type = kFloat)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API std::ostream& operator<<(std::ostream& strm, Tensor t);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor square(Tensor a);
TINYTORCH_API Tensor operator+(Tensor a, Tensor b);
inline Tensor operator+(double a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator+(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API Tensor operator-(Tensor a, Tensor b);
inline Tensor operator-(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator-(double a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API Tensor operator*(Tensor a, Tensor b);
inline Tensor operator*(double a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator*(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator/(Tensor a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator/(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator-(Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
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
TINYTORCH_API Tensor sum(Tensor a);

TINYTORCH_API Tensor& operator+=(Tensor& a, Tensor b);
TINYTORCH_API Tensor& operator-=(Tensor& a, Tensor b);
TINYTORCH_API Tensor& operator*=(Tensor& a, Tensor b);
TINYTORCH_API Tensor& operator/=(Tensor& a, Tensor b);


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

// Internal implementation of forward/backward
// Should NOT be called by the user
void fill_impl(Tensor a, float value);
Tensor square_impl(Tensor a);
Tensor add_impl(Tensor a, Tensor b);
Tensor add_impl(Tensor a, double b);
Tensor add_impl(double a, Tensor b);
Tensor sub_impl(Tensor a, Tensor b);
Tensor sub_impl(Tensor a, double b);
Tensor sub_impl(double a, Tensor b);
Tensor mult_impl(Tensor a, Tensor b);
Tensor mult_impl(Tensor a, double b);
Tensor mult_impl(double a, Tensor b);
Tensor div_impl(Tensor a, Tensor b);
Tensor div_impl(Tensor a, double b);
Tensor neg_impl(Tensor a);
Tensor sum_impl(Tensor a);
Tensor log_impl(Tensor a);
Tensor log1p_impl(Tensor a);
Tensor exp_impl(Tensor a);
Tensor sin_impl(Tensor a);
Tensor cos_impl(Tensor a);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output);
std::vector<Tensor> add_backward_impl(Tensor grad_output);
std::vector<Tensor> sub_backward_impl(Tensor grad_output);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output);
std::vector<Tensor> mult_backward_impl(Tensor a, double b, Tensor grad_output); // Returns only one gradient, the one for the tensor.
std::vector<Tensor> mult_backward_impl(double b, Tensor a, Tensor grad_output); // Returns only one gradient, the one for the tensor.
std::vector<Tensor> div_backward_impl(Tensor a, Tensor b, Tensor grad_output);
std::vector<Tensor> div_backward_impl(Tensor a, double b, Tensor grad_output); // Returns only one gradient, the one for the tensor.
std::vector<Tensor> neg_backward_impl(Tensor grad_output);
std::vector<Tensor> sum_backward_impl(const std::vector<int64_t>& input_sizes, Tensor grad_output);
std::vector<Tensor> log_backward_impl(Tensor a, Tensor grad_output);
std::vector<Tensor> log1p_backward_impl(Tensor a, Tensor grad_output);
std::vector<Tensor> exp_backward_impl(Tensor a, Tensor grad_output);
std::vector<Tensor> sin_backward_impl(Tensor a, Tensor grad_output);
std::vector<Tensor> cos_backward_impl(Tensor a, Tensor grad_output);

}  // namespace TINY_TORCH_NAMESPACE