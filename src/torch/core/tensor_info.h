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


#define MAX_TENSORINFO_DIMS 8

template <typename T>
inline std::pair<int64_t, int64_t> collapse_dims(T* sizes, T* strides, int64_t dims, const int excludeDim = -1)
{
    CHECK(excludeDim >= -1 && excludeDim < dims) << "expected excluded dim between -1 and dims - 1";

    int64_t stopDim             = (excludeDim == -1) ? dims : excludeDim;
    int64_t newIndex            = -1;
    int64_t oldIndex            = 0;
    int64_t remappedExcludedDim = -1;

    while (oldIndex < dims)
    {
        // Finds a dimension to collapse into
        for (; oldIndex < stopDim; ++oldIndex)
        {
            if (sizes[oldIndex] == 1)
            {
                continue;
            }

            ++newIndex;
            sizes[newIndex]   = sizes[oldIndex];
            strides[newIndex] = strides[oldIndex];
            ++oldIndex;
            break;
        }

        // Collapses dims
        for (; oldIndex < stopDim; ++oldIndex)
        {
            if (sizes[oldIndex] == 1)
            {
                continue;
            }

            if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex])
            {
                sizes[newIndex] *= sizes[oldIndex];
                strides[newIndex] = strides[oldIndex];
            }
            else
            {
                ++newIndex;
                sizes[newIndex]   = sizes[oldIndex];
                strides[newIndex] = strides[oldIndex];
            }
        }

        // Handles excludeDim being set (oldIndex == excludeDim)
        if (oldIndex != dims)
        {
            // Preserves excluded dimension
            ++newIndex;
            sizes[newIndex]     = sizes[oldIndex];
            strides[newIndex]   = strides[oldIndex];
            remappedExcludedDim = newIndex;

            // Restarts iteration after excludeDim
            ++oldIndex;
            stopDim = dims;
        }
    }

    // Handles special case of all dims size 1
    if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1))
    {
        dims       = 1;
        sizes[0]   = 1;
        strides[0] = 1;

        return std::pair<int64_t, int64_t>(0, 1);
    }

    dims = newIndex + 1;
    return std::pair<int64_t, int64_t>(remappedExcludedDim, dims);
}

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

    constexpr TT_HD void set_index(int64_t dim, int64_t value)
    {
#pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
            if (i == dim)
            {
                indices[i] = value;
            }
        }
    }

    constexpr TT_HD int64_t get_index(int64_t dim)
    {
#pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
            if (i == dim)
            {
                return indices[i];
            }
        }
        return 0;
    }

    constexpr TT_HD int64_t& operator[](int64_t dim)
    {
#if 1 || defined(__CUDACC__)
#    pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
            if (i == dim)
            {
                return indices[i];
            }
        }
        return indices[0];
#else
        return indices[dim];
#endif
    }

    constexpr TT_HD int64_t operator[](int64_t dim) const
    {
#if 1 || defined(__CUDACC__)
#    pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
            if (i == dim)
            {
                return indices[i];
            }
        }
        return 0;
#else
        return indices[dim];
#endif
    }

    TT_HD void zero_()
    {
#pragma unroll
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

    TensorInfoBase()
    {
        data  = nullptr;
        dims_ = 0;
    }
    // TensorInfoBase(T* p, int dim, int64_t sz[max_dims], int64_t st[max_dims]);
    TensorInfoBase(Tensor t)
    {
        if (!t.defined())
        {
            data = nullptr;
            return;
        }
        data  = t.template data_ptr<T>();
        dims_ = t.dim();
        CHECK_LE(t.dim(), MAX_TENSORINFO_DIMS);
        for (int i = 0; i < max_dims; ++i)
        {
            if (i < t.dim())
            {
                sizes[i]   = t.size(i);
                strides[i] = t.stride(i);
            }
            else
            {
                sizes[i]   = 1;
                strides[i] = 1;
            }
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

    TT_HD int64_t dim()
    {
        if constexpr (is_dynamic)
        {
            return dims_;
        }
        else
        {
            return max_dims;
        }
    }
    TT_HD int64_t size(int index) { return sizes[index]; }

    TT_HD int64_t numel()
    {
        int64_t result = 1;
#pragma unroll
        for (int64_t i = 0; i < max_dims; ++i)
        {
            if (i < dim())
            {
                result *= sizes[i];
            }
        }
        return result;
    }


    TT_HD T& operator[](int64_t linearId)
    {
        if (contiguous)
        {
            return data[linearId];
        }
        return operator[](LinearIndexToDimIndex(linearId));
    }
    TT_HD T& operator[](DimIndex index) { return data[IndexToOffset(index)]; }


    template <typename... Ts>
    TT_HD inline T& operator()(Ts... args)
    {
        return operator[](IndexToOffset(DimIndex({args...})));
    }

    TT_HD int64_t IndexToOffset(DimIndex index)
    {
        int64_t offset = 0;

#pragma unroll
        for (int64_t i = 0; i < max_dims; ++i)
        {
            if (i < dim())
            {
#if TT_DEBUG
#    if defined(__CUDACC__)
                CUDA_KERNEL_ASSERT(index[i] >= 0);
                CUDA_KERNEL_ASSERT(index[i] < sizes[i]);

#    else
                CHECK_GE(index[i], 0);
                CHECK_LT(index[i], sizes[i]);
#    endif
#endif
                offset += index[i] * strides[i];
            }
        }

        return offset;
    }

    TT_HD DimIndex LinearIndexToDimIndex(int64_t linearId)
    {
        DimIndex result;

#pragma unroll
        // for (int64_t i = max_dims - 1; i > 0; --i)
        for (int64_t j = 0; j < max_dims - 1; ++j)
        {
            auto i = max_dims - j - 1;
            // if(i==dim()) break;
            if (i < dim())
            {
                int64_t curDimIndex = linearId % sizes[i];
                result[i]           = curDimIndex;
                linearId /= sizes[i];
            }
        }

        result[0] = linearId;
        return result;
    }

    TT_HD DimIndex LinearIndexToDimIndexSkipDim(int64_t linearId, int64_t dim_to_skip)
    {
        DimIndex result;

#pragma unroll
        // for (int64_t i = max_dims - 1; i > 0; --i)
        for (int64_t j = 0; j < max_dims - 1; ++j)
        {
            auto i = max_dims - j - 1;
            if (i < dim())
            {
                int64_t curDimIndex = linearId % sizes[i];
                int64_t siz         = sizes[i];

                if (i == dim_to_skip)
                {
                    curDimIndex = 0;
                    siz         = 1;
                }

                result[i] = curDimIndex;
                linearId /= siz;
            }
        }

        result[0] = linearId;
        return result;
    }
    TT_HD bool index_in_range(DimIndex index)
    {
        bool result = true;
#pragma unroll
        for (int64_t i = 0; i < max_dims; ++i)
        {
            if (i < dim() && (index[i] < 0 || index[i] > sizes[i]))
            {
                result = false;
            }
        }
        return result;
    }

    TT_HD DimIndex clamp_index_to_size(DimIndex index)
    {
        DimIndex result;

#pragma unroll
        for (int64_t i = 0; i < max_dims; ++i)
        {
            if (i < dim())
            {
                result[i] = std::clamp(index[i], int64_t(0), sizes[i] - 1);
            }
        }
        return result;
    }


    T* data       = nullptr;
    int64_t dims_ = 0;
    int64_t sizes[max_dims];
    int64_t strides[max_dims];
    bool contiguous = false;
};

// template <typename T, bool TIS_CUDA, int MAX_DIMS>
// TensorInfoBase<T, TIS_CUDA, MAX_DIMS>::TensorInfoBase(T* p, int dim, int64_t sz[max_dims], int64_t st[max_dims])
//{
//     data  = p;
//     dims_ = dim;
//
//     for (int i = 0; i < dim; ++i)
//     {
//         sizes[i]   = sz[i];
//         strides[i] = st[i];
//     }
//
//     contiguous              = true;
//     int64_t expected_stride = 1;
//     for (int64_t i = dim - 1; i >= 0; --i)
//     {
//         if (strides[i] != expected_stride)
//         {
//             contiguous = false;
//             break;
//         }
//         expected_stride *= sizes[i];
//     }
// }

template <typename T, int MAX_DIMS = -1>
using TensorInfo = TensorInfoBase<T, false, MAX_DIMS>;


template <typename T, int MAX_DIMS = -1>
using TensorInfoCuda = TensorInfoBase<T, true, MAX_DIMS>;


}  // namespace tinytorch
