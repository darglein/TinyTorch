/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "conv.h"

#include "torch/cuda/ops_impl_cuda_helper.h"

namespace tinytorch
{
namespace cuda_impl
{

template <typename T>
__launch_bounds__(128) static __global__
    void conv2d_impl_kernel(TensorInfoCuda<T, 4> input, TensorInfoCuda<T, 4> weight, TensorInfoCuda<T, 4> bias,
                            int stride, int padding, int dilation, int groups, TensorInfoCuda<T, 4> result)
{
    // NOTE: mirrors cpu_impl::conv2d exactly (stride/padding/dilation/groups are accepted but
    // not used; border clipping is applied; weight is {1,1,kH,kW} shared across channels)
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    auto index_result = result.LinearIndexToDimIndex(i);

    T sum = T(0.);
    for (int64_t j = 0; j < weight.size(2); ++j)
    {
        for (int64_t k = 0; k < weight.size(3); ++k)
        {
            auto index_weight = index_result;
            index_weight[2]   = j;
            index_weight[3]   = k;
            auto w            = weight(index_weight);

            auto index_input = index_result;
            index_input[2] += j - (weight.size(2) / 2);
            index_input[3] += k - (weight.size(3) / 2);

            index_input[2] = std::min(index_input[2], (int64_t)input.size(2) - 1);
            index_input[3] = std::min(index_input[3], (int64_t)input.size(3) - 1);
            index_input[2] = std::max(index_input[2], (int64_t)0);
            index_input[3] = std::max(index_input[3], (int64_t)0);

            auto v     = input[index_input];
            sum        = sum + w * v;
        }
    }
    result[index_result] = sum;
}

void conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups, Tensor result)
{
    cuda::DeviceGuard guard(input.device());
    CUDA_SWITCH_MACRO_FLOAT(input.device(), input.scalar_type(), result.numel(), conv2d_impl_kernel, input, weight,
                            bias, stride, padding, dilation, groups, result);
}
}  // namespace cuda_impl
}  // namespace tinytorch
