
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
Tensor square_impl_cuda(Tensor a);
Tensor sum_impl_cuda(Tensor a);
Tensor log_impl_cuda(Tensor a);
Tensor log1p_impl_cuda(Tensor a);
Tensor exp_impl_cuda(Tensor a);
Tensor sign_impl_cuda(Tensor a);
Tensor pow_impl_cuda(Tensor a, double b);
Tensor sin_impl_cuda(Tensor a);
Tensor cos_impl_cuda(Tensor a);
Tensor relu_impl_cuda(Tensor a);
Tensor sigmoid_impl_cuda(Tensor a);
Tensor softplus_impl_cuda(Tensor a, double beta);
//Tensor prod_impl_cuda(Tensor a, int64_t dim);
//Tensor min_impl_cuda(Tensor a);
//Tensor max_impl_cuda(Tensor a);
Tensor min_impl_cuda(Tensor a, Tensor b);
Tensor max_impl_cuda(Tensor a, Tensor b);

void copy_impl_cuda(Tensor src, Tensor target);

}


#endif