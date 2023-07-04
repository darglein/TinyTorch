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
TINYTORCH_API void manual_seed(int64_t seed);
TINYTORCH_API int64_t get_seed();

// Basic tensor generation
TINYTORCH_API Tensor full(const SizeType& sizes, float value, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor ones(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor empty(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor zeros(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor rand(const SizeType& sizes, TensorOptions options = TensorOptions());
TINYTORCH_API Tensor randint(int low, int high, const SizeType& sizes,
                             TensorOptions options = TensorOptions());

TINYTORCH_API Tensor full_like(Tensor t, float value);
TINYTORCH_API Tensor ones_like(Tensor t);
TINYTORCH_API Tensor empty_like(Tensor t);
TINYTORCH_API Tensor zeros_like(Tensor t);
TINYTORCH_API Tensor rand_like(Tensor t);

inline Tensor empty_like(Tensor t, TensorOptions options){
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor ones_like(Tensor t, TensorOptions options){
    throw std::runtime_error("not implemented");
    return {};
}

TINYTORCH_API void fill(Tensor& t, double value);

inline std::pair<Tensor,Tensor> sort(Tensor t, int64_t a)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor range(int64_t start, int64_t end, int64_t a)
{
    throw std::runtime_error("not implemented");
    return {};
}

inline Tensor range(int64_t start, int64_t end, TensorOptions options)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor from_blob(void* data, const SizeType& sizes, const SizeType& stride,
                        ScalarType type = kFloat)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor from_blob(void* data, const SizeType& sizes, ScalarType type = kFloat)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor from_blob(void* data, const SizeType& sizes, TensorOptions options)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API std::ostream& operator<<(std::ostream& strm, Tensor t);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor square(Tensor a);
TINYTORCH_API Tensor operator+(Tensor a, Tensor b);

TINYTORCH_API Tensor operator+(Tensor a, double b);
inline Tensor operator+(double a, Tensor b)
{
   return b + a;
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
TINYTORCH_API Tensor operator*(double a, Tensor b);
inline Tensor operator*(Tensor a, double b)
{
    return b * a;
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
inline Tensor operator/(double a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
TINYTORCH_API Tensor operator-(Tensor b);
inline Tensor operator==(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator<(Tensor a, double b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor operator>(Tensor a, double b)
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
inline Tensor min(Tensor a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
inline Tensor max(Tensor a, Tensor b)
{
    throw std::runtime_error("not implemented");
    return {};
}
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
TINYTORCH_API Tensor sum(Tensor a);

TINYTORCH_API Tensor operator+=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator+=(Tensor a, double b);
TINYTORCH_API Tensor operator-=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator-=(Tensor a, double b);
TINYTORCH_API Tensor operator*=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator*=(Tensor a, double b);
TINYTORCH_API Tensor operator/=(Tensor a, Tensor b);
TINYTORCH_API Tensor operator/=(Tensor a, double b);
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