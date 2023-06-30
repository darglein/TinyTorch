/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "assert.h"
#include "torch/core/tensor.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tensor_data.h"
#include "torch/tiny_torch_config.h"
#include <type_traits>

namespace TINY_TORCH_NAMESPACE
{


#define MAX_TENSORINFO_DIMS 25

// CUDA kernel argument that defines tensor layout
template <typename T, int MAX_DIMS = -1>
struct TensorInfo
{
    static constexpr bool is_dynamic = MAX_DIMS == -1;
    static constexpr int max_dims    = is_dynamic ? MAX_TENSORINFO_DIMS : MAX_DIMS;

    TensorInfo();
    TensorInfo(T* p, int dim, int64_t sz[max_dims], int64_t st[max_dims]);
    TensorInfo(Tensor t)
    {
        data = t.template data_ptr<T>();
        dims = t.dim();
        for (int i = 0; i < t.dim(); ++i)
        {
            sizes[i]   = t.size(i);
            strides[i] = t.stride(i);
        }
    }

    int64_t numel()
    {
        int64_t result = 1;
        for (int64_t i = 0; i < dims; ++i)
        {
            result *= sizes[i];
        }
        return result;
    }

    T& GetLinear(int64_t linearId) { return data[IndexToOffset(linearId)]; }

    T& operator[](int64_t linearId) { return data[IndexToOffset(linearId)]; }


    int64_t IndexToOffset(int64_t linearId)
    {
        if constexpr (is_dynamic)
        {
            int64_t offset = 0;

            for (int64_t i = dims - 1; i > 0; --i)
            {
                int64_t curDimIndex  = linearId % sizes[i];
                int64_t curDimOffset = curDimIndex * strides[i];
                offset += curDimOffset;
                linearId /= sizes[i];
            }

            return offset + linearId * strides[0];
        }
        else
        {
            int64_t offset = 0;

            // Uses static dims
            for (int64_t i = max_dims - 1; i > 0; --i)
            {
                int64_t curDimIndex  = linearId % sizes[i];
                int64_t curDimOffset = curDimIndex * strides[i];
                offset += curDimOffset;
                linearId /= sizes[i];
            }

            return offset + linearId * strides[0];
        }
    }

    // Contiguous tensors of more than one dimension are collapsed down
    // to one tensor
    inline bool isContiguous() const { return (dims == 1 && strides[0] == 1); }

    T* data;
    int64_t sizes[max_dims];
    int64_t strides[max_dims];
    int64_t dims;
};


template <typename T, int MAX_DIMS>
TensorInfo<T, MAX_DIMS>::TensorInfo()
{
    data = nullptr;
    dims = 0;
}

template <typename T, int MAX_DIMS>
TensorInfo<T, MAX_DIMS>::TensorInfo(T* p, int dim, int64_t sz[max_dims], int64_t st[max_dims])
{
    data = p;
    dims = dim;

    for (int i = 0; i < dim; ++i)
    {
        sizes[i]   = sz[i];
        strides[i] = st[i];
    }
}



}  // namespace TINY_TORCH_NAMESPACE
