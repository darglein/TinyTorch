/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/core/ops_functions.h"

#include "graph.h"

#include "torch/cpu/ops_impl_cpu.h"


namespace tinytorch
{

Tensor repeat(Tensor t, SizeType sizes)
{
    CHECK_EQ(t.dim(), sizes.size());

    int repeat_dim = -1;
    for (int i = 0; i < sizes.size(); ++i)
    {
        if (sizes[i] > 1)
        {
            CHECK_EQ(repeat_dim, -1);
            repeat_dim = i;
        }
    }


    auto new_size        = t.sizes().vec();
    new_size[repeat_dim] = new_size[repeat_dim] * sizes[repeat_dim];


    Tensor result = empty(new_size, t.options());

    for (int i = 0; i < sizes[repeat_dim]; ++i)
    {
        result.slice(repeat_dim, i * t.size(repeat_dim), (i + 1) * t.size(repeat_dim)).copy_(t);
    }

    //     std::cout << "new_size " << new_size << std::endl;
    // throw std::runtime_error("lsdf");
    return result;
}

void fill(Tensor& t, double value)
{
    fill_impl_cpu(t, value);
}

Tensor index_select(Tensor input, int64_t dim, Tensor index)
{
    return index_select_impl_cpu(input, dim, index);
}


}  // namespace tinytorch