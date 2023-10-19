/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "conv.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops/ops_impl.h"
#include "torch/core/tensor_info.h"
namespace tinytorch
{
namespace cpu_impl
{



template <typename T>
static void conv2d_impl(TensorInfo<T, 4> input, TensorInfo<T, 4> weight, TensorInfo<T, 4> bias, int stride, int padding,
                        int dilation, int groups, TensorInfo<T, 4> result)
{
    using IndexType = typename TensorInfo<T, 4>::IndexType;
        CHECK_EQ(weight.size(0), 1);
    CHECK_EQ(weight.size(1), 1);
    for (int64_t i = 0; i < result.numel(); ++i)
    {
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

                index_input[2] = std::min(index_input[2], input.size(2) - 1);
                index_input[3] = std::min(index_input[3], input.size(3) - 1);
                index_input[2] = std::max(index_input[2], IndexType(0));
                index_input[3] = std::max(index_input[3], IndexType(0));

                auto v = input[index_input];

                sum = sum + w * v;
            }
        }
        result[index_result] = sum;
    }
}

void conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups, Tensor result)
{
//    std::cout << "conv" << std::endl;
//    std::cout << input.sizes() << std::endl;
//    std::cout << weight.sizes() << std::endl;
//    std::cout << result.sizes() << std::endl;
    SWITCH_MACRO_FLOAT(input.scalar_type(), conv2d_impl, input, weight, bias, stride, padding, dilation, groups,
                       result);
//    exit(0);
}
}  // namespace cpu_impl

}  // namespace tinytorch
