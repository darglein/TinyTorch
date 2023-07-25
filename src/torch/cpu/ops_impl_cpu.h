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
void sqrt_impl_cpu(Tensor a, Tensor& result);
void sum_impl_cpu(Tensor a, Tensor& result);
void sum_impl_cpu(Tensor a, int64_t dim, Tensor& result);
void log_impl_cpu(Tensor a, Tensor& result);
void log1p_impl_cpu(Tensor a, Tensor& result);
void exp_impl_cpu(Tensor a, Tensor& result);
void sign_impl_cpu(Tensor a, Tensor& result);
void pow_impl_cpu(Tensor a, double b, Tensor& result);
void sin_impl_cpu(Tensor a, Tensor& result);
void cos_impl_cpu(Tensor a, Tensor& result);
void relu_impl_cpu(Tensor a, Tensor& result);
void sigmoid_impl_cpu(Tensor a, Tensor& result);
void softplus_impl_cpu(Tensor a, double beta, Tensor& result);
void prod_impl_cpu(Tensor a, int64_t dim, Tensor& result);
void min_impl_cpu(Tensor a, Tensor& result);
void max_impl_cpu(Tensor a, Tensor& result);
void min_impl_cpu(Tensor a, Tensor b, Tensor& result);
void max_impl_cpu(Tensor a, Tensor b, Tensor& result);
void min_impl_cpu(Tensor a, int64_t dim, bool keepdim, Tensor& result, Tensor& indices);
void max_impl_cpu(Tensor a, int64_t dim, bool keepdim, Tensor& result, Tensor& indices);
void std_impl_cpu(Tensor a, Tensor& result);
void abs_impl_cpu(Tensor a, Tensor& result);
void index_select_impl_cpu(Tensor input, int64_t dim, Tensor index, Tensor& result);
void index_add_impl_cpu(Tensor input, int64_t dim, Tensor index, Tensor data, Tensor& result);
void repeat_interleave_impl_cpu(Tensor input, int64_t count, Tensor& result);
void stack_impl_cpu(const std::vector<Tensor>& tensors, Tensor& result);
void transpose_impl_cpu(Tensor input, int64_t dim0, int64_t dim1, Tensor& result);
void copy_and_convert_impl_cpu(Tensor src, Tensor& target);
void clamp_impl_cpu_(Tensor& a, double low, double high);

void sum_backward_impl_cpu(Tensor grad_output, Tensor& grad_a);
void log_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void log1p_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void exp_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void sign_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void pow_backward_impl_cpu(Tensor a, double b, Tensor grad_output, Tensor& grad_a);
void sin_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void cos_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void relu_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void sigmoid_backward_impl_cpu(Tensor a, Tensor grad_output, Tensor& grad_a);
void softplus_backward_impl_cpu(Tensor a, double beta, Tensor grad_output, Tensor& grad_a);
void prod_backward_impl_cpu(Tensor a, int64_t dim, Tensor grad_output, Tensor& grad_a);
void min_backward_impl_cpu(Tensor grad_output, Tensor& grad_a, Tensor& grad_b);
void max_backward_impl_cpu(Tensor grad_output, Tensor& grad_a, Tensor& grad_b);

}  // namespace tinytorch