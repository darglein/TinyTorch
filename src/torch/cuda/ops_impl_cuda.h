
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

void sum_impl(Tensor a, Tensor& result);
void sum_impl(Tensor a, int64_t dim, Tensor& result);


void min_impl(Tensor a, Tensor b, Tensor& result);

void max_impl(Tensor a,  Tensor& result);
void max_impl(Tensor a, Tensor b, Tensor& result);

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor& result);
void index_add_impl( int64_t dim, Tensor index, Tensor data, Tensor& result);
void repeat_interleave_impl(Tensor input, int64_t count, Tensor& result);
void transpose_impl(Tensor input, int64_t dim0, int64_t dim1, Tensor& result);

}  // namespace cuda_impl
}  // namespace tinytorch


#endif