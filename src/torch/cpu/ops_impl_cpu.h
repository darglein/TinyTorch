/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/tensor.h"

#include "torch/core/tensor_options.h"
#include "torch/tiny_torch_config.h"
namespace tinytorch
{
// Internal implementation of forward/backward
// Should NOT be called by the user
void range_impl_cpu(Tensor a, double start, double end, double step);
void fill_impl_cpu(Tensor a, double value);
Tensor square_impl_cpu(Tensor a);
Tensor add_impl_cpu(Tensor a, Tensor b);
Tensor add_impl_cpu(Tensor a, double b);
Tensor add_impl_cpu(double a, Tensor b);
Tensor sub_impl_cpu(Tensor a, Tensor b);
Tensor sub_impl_cpu(Tensor a, double b);
Tensor sub_impl_cpu(double a, Tensor b);
Tensor mult_impl_cpu(Tensor a, Tensor b);
Tensor mult_impl_cpu(Tensor a, double b);
Tensor mult_impl_cpu(double a, Tensor b);
Tensor div_impl_cpu(Tensor a, Tensor b);
Tensor div_impl_cpu(Tensor a, double b);
Tensor div_impl_cpu(double a, Tensor b);
Tensor neg_impl_cpu(Tensor a);
Tensor sum_impl_cpu(Tensor a);
Tensor log_impl_cpu(Tensor a);
Tensor log1p_impl_cpu(Tensor a);
Tensor exp_impl_cpu(Tensor a);
Tensor sign_impl_cpu(Tensor a);
Tensor pow_impl_cpu(Tensor a, double b);
Tensor sin_impl_cpu(Tensor a);
Tensor cos_impl_cpu(Tensor a);
Tensor relu_impl_cpu(Tensor a);
Tensor sigmoid_impl_cpu(Tensor a);
Tensor softplus_impl_cpu(Tensor a, double beta);
Tensor prod_impl_cpu(Tensor a, int64_t dim);
Tensor min_impl_cpu(Tensor a);
Tensor max_impl_cpu(Tensor a);
Tensor min_impl_cpu(Tensor a, Tensor b);
Tensor max_impl_cpu(Tensor a, Tensor b);
std::pair<Tensor, Tensor> min_impl_cpu(Tensor a, int64_t dim, bool keepdim);
std::pair<Tensor, Tensor> max_impl_cpu(Tensor a, int64_t dim, bool keepdim);
Tensor std_impl_cpu(Tensor a);
Tensor index_select_impl_cpu(Tensor input, int64_t dim, Tensor index);
Tensor repeat_interleave_impl_cpu(Tensor input, int64_t count);
Tensor stack_impl_cpu(const std::vector<Tensor>& tensors);
Tensor transpose_impl_cpu(Tensor input, int64_t dim0, int64_t dim1);
std::vector<Tensor> square_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> add_backward_impl_cpu(Tensor grad_output);
std::vector<Tensor> sub_backward_impl_cpu(Tensor grad_output);
std::vector<Tensor> mult_backward_impl_cpu(Tensor a, Tensor b, Tensor grad_output);
std::vector<Tensor> mult_backward_impl_cpu(Tensor a, double b, Tensor grad_output); // Returns only one gradient, the one for the tensor.
std::vector<Tensor> mult_backward_impl_cpu(double b, Tensor a, Tensor grad_output); // Returns only one gradient, the one for the tensor.
std::vector<Tensor> div_backward_impl_cpu(Tensor a, Tensor b, Tensor grad_output);
std::vector<Tensor> div_backward_impl_cpu(Tensor a, double b, Tensor grad_output); // Returns only one gradient, the one for the tensor.
std::vector<Tensor> div_backward_impl_cpu(double a, Tensor b, Tensor grad_output); // Returns only one gradient, the one for the tensor.
std::vector<Tensor> neg_backward_impl_cpu(Tensor grad_output);
std::vector<Tensor> sum_backward_impl_cpu(const SizeType& input_sizes, Tensor grad_output);
std::vector<Tensor> log_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> log1p_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> exp_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> sign_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> pow_backward_impl_cpu(Tensor a, double b, Tensor grad_output);
std::vector<Tensor> sin_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> cos_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> relu_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> sigmoid_backward_impl_cpu(Tensor a, Tensor grad_output);
std::vector<Tensor> softplus_backward_impl_cpu(Tensor a, double beta, Tensor grad_output);
std::vector<Tensor> prod_backward_impl_cpu(Tensor a, int64_t dim, Tensor grad_output);
std::vector<Tensor> min_backward_impl_cpu(Tensor grad_output);
std::vector<Tensor> max_backward_impl_cpu(Tensor grad_output);

}  // namespace tinytorch