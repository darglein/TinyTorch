/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */

#pragma once
#include "tiny_torch_config.h"
#include "tensor.h"
namespace tinytorch
{

// Basic tensor generation
TINYTORCH_API Tensor zero(int size);
TINYTORCH_API Tensor rand(int size);
TINYTORCH_API std::ostream& operator<<(std::ostream& strm, Tensor t);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
TINYTORCH_API Tensor square(Tensor a);
TINYTORCH_API Tensor operator-(Tensor a, Tensor b);
TINYTORCH_API Tensor operator+(Tensor a, Tensor b);
TINYTORCH_API Tensor operator*(Tensor a, Tensor b);
TINYTORCH_API Tensor sum(Tensor a);


// Internal implementation of forward/backward
// Should NOT be called by the user
Tensor square_impl(Tensor a);
Tensor sub_impl(Tensor a, Tensor b);
Tensor add_impl(Tensor a, Tensor b);
Tensor mult_impl(Tensor a, Tensor b);
Tensor sum_impl(Tensor a);
std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output);
std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output);
std::vector<Tensor>  add_backward_impl(Tensor grad_output);
std::vector<Tensor>  sub_backward_impl(Tensor grad_output);
std::vector<Tensor>  sum_backward_impl(int input_size, Tensor grad_output);

}  // namespace tinytorch