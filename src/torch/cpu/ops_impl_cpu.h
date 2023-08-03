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
namespace cpu_impl
{
void print_impl(std::ostream& strm, Tensor t);
void to_impl_cpu_cuda(Tensor a, Tensor b);

void range_impl(Tensor a, double start, double end, double step);

void fill_impl(Tensor& a, double value);
void fill_impl(Tensor& a, Tensor value);
void fill_impl(Tensor& a, Tensor values, int dim);
void permute_impl(Tensor& src, Tensor& result, SizeType new_dims);

void uniform_impl(Tensor& t, double mi, double ma);
void uniform_int_impl(Tensor& t, int low, int high);
void sum_impl(Tensor a, Tensor& result);
void sum_impl(Tensor a, int64_t dim, Tensor& result);


void pow_impl(Tensor a, double b, Tensor& result);

void prod_impl(Tensor a, int64_t dim, Tensor& result);
void min_impl(Tensor a, Tensor& result);
void min_impl(Tensor a, Tensor b, Tensor& result);
void min_impl(Tensor a, int64_t dim, bool keepdim, Tensor& result, Tensor& indices);
void max_impl(Tensor a, Tensor& result);
void max_impl(Tensor a, Tensor b, Tensor& result);
void max_impl(Tensor a, int64_t dim, bool keepdim, Tensor& result, Tensor& indices);

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor& result);
void index_add_impl( int64_t dim, Tensor index, Tensor data, Tensor& result);
void repeat_interleave_impl(Tensor input, int64_t count, Tensor& result);
void transpose_impl(Tensor input, int64_t dim0, int64_t dim1, Tensor& result);
void copy_and_convert_impl(Tensor src, Tensor& target);
void clamp_impl_(Tensor& a, double low, double high);

void sum_backward_impl(Tensor grad_output, Tensor& grad_a);

void prod_backward_impl(Tensor a, int64_t dim, Tensor grad_output, Tensor& grad_a);
void min_backward_impl(Tensor grad_output, Tensor& grad_a, Tensor& grad_b);
void max_backward_impl(Tensor grad_output, Tensor& grad_a, Tensor& grad_b);

}  // namespace cpu_impl
}  // namespace tinytorch