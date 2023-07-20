
#pragma once
#include "torch/core/tensor.h"

#include "torch/core/tensor_options.h"
#include "torch/tiny_torch_config.h"


#ifdef TT_HAS_CUDA

namespace tinytorch
{
// Internal implementation of forward/backward
// Should NOT be called by the user
void range_impl_cuda(Tensor a, double start, double end, double step);
void fill_impl_cuda(Tensor a, double value);
void copy_and_convert_impl_cuda(Tensor src, Tensor& target);
void uniform_impl_cuda(Tensor& t, double mi, double ma);
void uniform_int_impl_cuda(Tensor& t, int low, int high);

void sum_impl_cuda(Tensor a, Tensor& result);
void log_impl_cuda(Tensor a, Tensor& result);
void log1p_impl_cuda(Tensor a, Tensor& result);
void exp_impl_cuda(Tensor a, Tensor& result);
void sign_impl_cuda(Tensor a, Tensor& result);
void pow_impl_cuda(Tensor a, double b, Tensor& result);
void sin_impl_cuda(Tensor a, Tensor& result);
void cos_impl_cuda(Tensor a, Tensor& result);
void relu_impl_cuda(Tensor a, Tensor& result);
void sigmoid_impl_cuda(Tensor a, Tensor& result);
void softplus_impl_cuda(Tensor a, double beta, Tensor& result);
//void prod_impl_cuda(Tensor a, int64_t dim, Tensor& result);
//void min_impl_cuda(Tensor a, Tensor& result);
//void max_impl_cuda(Tensor a, Tensor& result);
void min_impl_cuda(Tensor a, Tensor b, Tensor& result);
void max_impl_cuda(Tensor a, Tensor b, Tensor& result);

}


#endif