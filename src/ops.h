/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
 */

#pragma once

#include "tensor.h"
namespace tinytorch
{

// Basic tensor generation
Tensor zero(int size);
Tensor rand(int size);
std::ostream& operator<<(std::ostream& strm, Tensor t);

// Basic Tensor Math operators
// These operators should be called by the user and support Auto-Diff
Tensor square(Tensor a);
Tensor operator-(Tensor a, Tensor b);
Tensor operator+(Tensor a, Tensor b);
Tensor operator*(Tensor a, Tensor b);
Tensor sum(Tensor a);


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