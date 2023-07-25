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

namespace tinytorch
{


#define MAX_TENSORINFO_DIMS 25


template <int DIM>
struct DimIndexStruct
{
    int64_t indices[DIM];
    TT_HD int64_t& operator[](int64_t i) { return indices[i]; }

    TT_HD void zero_()
    {
        for (int i = 0; i < DIM; ++i)
        {
            indices[i] = 0;
        }
    }
};


// CUDA kernel argument that defines tensor layout
template <typename T, int MAX_DIMS = -1>
struct TensorInfo
{
    static constexpr bool is_dynamic = MAX_DIMS == -1;
    static constexpr int max_dims    = is_dynamic ? MAX_TENSORINFO_DIMS : MAX_DIMS;
    using DimIndex                   = DimIndexStruct<max_dims>;

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
        contiguous = t.is_contiguous();
    }

    TT_HD int64_t numel()
    {
        int64_t result = 1;
        for (int64_t i = 0; i < dims; ++i)
        {
            result *= sizes[i];
        }
        return result;
    }


    TT_HD T& operator[](int64_t linearId) { return operator[](LinearIndexToDimIndex(linearId)); }
    TT_HD T& operator[](DimIndex index) { return data[IndexToOffset(index)]; }

    TT_HD int64_t IndexToOffset(DimIndex index)
    {
        int64_t offset = 0;
        if constexpr (is_dynamic)
        {
            for (int64_t i = 0; i < dims; ++i)
            {
                offset += index[i] * strides[i];
            }
        }
        else
        {
            for (int64_t i = 0; i < max_dims; ++i)
            {
                offset += index[i] * strides[i];
            }
        }
        return offset;
    }

    TT_HD DimIndex LinearIndexToDimIndex(int64_t linearId)
    {
        DimIndex result;
        result.zero_();
        if constexpr (is_dynamic)
        {
            for (int64_t i = dims - 1; i > 0; --i)
            {
                int64_t curDimIndex = linearId % sizes[i];
                result[i]           = curDimIndex;
                linearId /= sizes[i];
            }

            result[0] = linearId;
        }
        else
        {
            // Uses static dims
            for (int64_t i = max_dims - 1; i > 0; --i)
            {
                int64_t curDimIndex = linearId % sizes[i];
                result[i]           = curDimIndex;
                linearId /= sizes[i];
            }
            result[0] = linearId;
        }
        return result;
    }


    DimIndex clamp_index_to_size(DimIndex index)
    {
        DimIndex result;
        for (int i = 0; i < max_dims; ++i)
        {
            result[i] = std::clamp(index[i], int64_t(0), sizes[i] - 1);
        }
        return result;
    }

    T* data;
    int64_t sizes[max_dims];
    int64_t strides[max_dims];
    int64_t dims;
    bool contiguous;
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

    contiguous              = true;
    int64_t expected_stride = 1;
    for (int64_t i = dim - 1; i >= 0; --i)
    {
        if (strides[i] != expected_stride)
        {
            contiguous = false;
            break;
        }
        expected_stride *= sizes[i];
    }
}



}  // namespace tinytorch
