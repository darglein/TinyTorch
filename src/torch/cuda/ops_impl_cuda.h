
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

void copy_impl_cuda(Tensor src, Tensor target);

}


#endif