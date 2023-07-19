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

void print_impl_cpu(std::ostream& strm, Tensor t);
void to_impl_cpu_cuda(Tensor a, Tensor b);

void range_impl_cpu(Tensor a, double start, double end, double step);
void fill_impl_cpu(Tensor a, double value);
void uniform_impl_cpu(Tensor& t, double mi, double ma);
void uniform_int_impl_cpu(Tensor& t, int low, int high);
Tensor square_impl_cpu(Tensor a);
void sqrt_impl_cpu(Tensor a, Tensor& result);
Tensor sum_impl_cpu(Tensor a);
void sum_impl_cpu(Tensor a, int64_t dim, Tensor& result);
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
Tensor abs_impl_cpu(Tensor a);
Tensor index_select_impl_cpu(Tensor input, int64_t dim, Tensor index);
Tensor index_add_impl_cpu(Tensor input, int64_t dim, Tensor index, Tensor data);
Tensor repeat_interleave_impl_cpu(Tensor input, int64_t count);
Tensor stack_impl_cpu(const std::vector<Tensor>& tensors);
Tensor transpose_impl_cpu(Tensor input, int64_t dim0, int64_t dim1);
Tensor to_impl_cpu(Tensor a, ScalarType other_type);
void copy_impl_cpu(Tensor src, Tensor target);
void clamp_impl_cpu_(Tensor& a, double low, double high);

std::vector<Tensor> square_backward_impl_cpu(Tensor a, Tensor grad_output);


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