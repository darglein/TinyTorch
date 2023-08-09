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
#include "torch/tiny_torch_cuda.h"
#include <type_traits>

namespace tinytorch
{


#define MAX_TENSORINFO_DIMS 25


template <int DIM>
struct DimIndexStruct
{
    int64_t indices[DIM];

    TT_HD DimIndexStruct() {}

    DimIndexStruct(const std::vector<int64_t>& data)
    {
        CHECK_LT(data.size(), DIM);
        for (int i = 0; i < data.size(); ++i)
        {
            indices[i] = data[i];
        }
    }


    TT_HD DimIndexStruct(std::initializer_list<int64_t> l)
    {
        int k = 0;
        for (auto i : l)
        {
            indices[k++] = i;
        }
    }

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
template <typename T, bool TIS_CUDA, int MAX_DIMS = -1>
struct TensorInfoBase
{
    static constexpr bool is_dynamic = MAX_DIMS == -1;
    static constexpr int max_dims    = is_dynamic ? MAX_TENSORINFO_DIMS : MAX_DIMS;
    using DimIndex                   = DimIndexStruct<max_dims>;

    TensorInfoBase();
    TensorInfoBase(T* p, int dim, int64_t sz[max_dims], int64_t st[max_dims]);
    TensorInfoBase(Tensor t)
    {
        data = t.template data_ptr<T>();
        dims = t.dim();
        for (int i = 0; i < t.dim(); ++i)
        {
            sizes[i]   = t.size(i);
            strides[i] = t.stride(i);
        }
        contiguous = t.is_contiguous();
        if (TIS_CUDA)
        {
            CHECK(t.is_cuda());
        }
        else
        {
            CHECK(t.is_cpu());
        }
    }

    TT_HD int64_t dim() { return dims; }
    TT_HD int64_t size(int index) { return sizes[index]; }

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


    template <typename... Ts>
    TT_HD inline T& operator()(Ts... args)
    {
        return operator[](IndexToOffset(DimIndex({args...})));
    }

    TT_HD int64_t IndexToOffset(DimIndex index)
    {
        int64_t offset = 0;
        if constexpr (is_dynamic)
        {
            for (int64_t i = 0; i < dims; ++i)
            {
#ifndef __CUDACC__
                CHECK_GE(index[i], 0);
                CHECK_LT(index[i], sizes[i]);
#else
                CUDA_KERNEL_ASSERT(index[i] >= 0);
                CUDA_KERNEL_ASSERT(index[i] < sizes[i]);
#endif
                offset += index[i] * strides[i];
            }
        }
        else
        {
            for (int64_t i = 0; i < max_dims; ++i)
            {
#ifndef __CUDACC__
                CHECK_GE(index[i], 0);
                CHECK_LT(index[i], sizes[i]);
#else
                CUDA_KERNEL_ASSERT(index[i] >= 0);
                CUDA_KERNEL_ASSERT(index[i] < sizes[i]);
#endif
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
    TT_HD bool index_in_range(DimIndex index)
    {
        bool result = true;
        for (int i = 0; i < dims; ++i)
        {
            if (index[i] < 0 || index[i] > sizes[i])
            {
                result = false;
            }
        }
        return result;
    }

    TT_HD DimIndex clamp_index_to_size(DimIndex index)
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


template <typename T, bool TIS_CUDA, int MAX_DIMS>
TensorInfoBase<T, TIS_CUDA, MAX_DIMS>::TensorInfoBase()
{
    data = nullptr;
    dims = 0;
}

template <typename T, bool TIS_CUDA, int MAX_DIMS>
TensorInfoBase<T, TIS_CUDA, MAX_DIMS>::TensorInfoBase(T* p, int dim, int64_t sz[max_dims], int64_t st[max_dims])
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

template <typename T, int MAX_DIMS = -1>
using TensorInfo = TensorInfoBase<T, false, MAX_DIMS>;


template <typename T, int MAX_DIMS = -1>
using TensorInfoCuda = TensorInfoBase<T, true, MAX_DIMS>;


}  // namespace tinytorch
