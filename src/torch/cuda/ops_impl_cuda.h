
#pragma once
#include "torch/core/tensor.h"

#include "torch/core/tensor_options.h"
#include "torch/tiny_torch_config.h"


#ifdef TT_HAS_CUDA

namespace tinytorch
{

namespace cuda_impl
{
// Internal implementation of forward/backward
// Should NOT be called by the user
void range_impl(Tensor a, double start, double end, double step);

void fill_impl(Tensor& a, double value);
void fill_impl(Tensor& a, Tensor value);
void fill_impl(Tensor& a, Tensor values, int dim);

void copy_and_convert_impl(Tensor src, Tensor& target);
void uniform_impl(Tensor& t, double mi, double ma);
void uniform_int_impl(Tensor& t, int low, int high);
void sqrt_impl(Tensor a, Tensor& result);
void sum_impl(Tensor a, Tensor& result);
void sum_impl(Tensor a, int64_t dim, Tensor& result);
void log_impl(Tensor a, Tensor& result);
void log1p_impl(Tensor a, Tensor& result);
void exp_impl(Tensor a, Tensor& result);
void sign_impl(Tensor a, Tensor& result);
void pow_impl(Tensor a, double b, Tensor& result);
void sin_impl(Tensor a, Tensor& result);
void cos_impl(Tensor a, Tensor& result);
void relu_impl(Tensor a, Tensor& result);
void sigmoid_impl(Tensor a, Tensor& result);
void softplus_impl(Tensor a, double beta, Tensor& result);
// void prod_impl(Tensor a, int64_t dim, Tensor& result);
// void min_impl(Tensor a, Tensor& result);
// void max_impl(Tensor a, Tensor& result);
void min_impl(Tensor a, Tensor b, Tensor& result);
void max_impl(Tensor a, Tensor b, Tensor& result);

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor& result);
void index_add_impl( int64_t dim, Tensor index, Tensor data, Tensor& result);

}  // namespace cuda_impl
}  // namespace tinytorch


#endif