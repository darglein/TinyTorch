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

#if defined(__CUDACC__)
#    define TT_INLINE __forceinline__
#else
#    define TT_INLINE inline
#endif

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

template <int DIM, typename IndexType>
struct DimIndexStruct
{
    IndexType indices[DIM];

    TT_HD DimIndexStruct() {}

    DimIndexStruct(const std::vector<int64_t>& data)
    {
        CHECK_LT(data.size(), DIM);
        for (int i = 0; i < data.size(); ++i)
        {
            indices[i] = (IndexType)data[i];
        }
    }


    TT_INLINE constexpr TT_HD DimIndexStruct(std::initializer_list<IndexType> l)
    {
        int k = 0;
        for (auto i : l)
        {
            indices[k++] = i;
        }
    }

    TT_INLINE constexpr TT_HD void set_index(IndexType dim, IndexType value)
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

    TT_INLINE constexpr TT_HD IndexType get_index(IndexType dim)
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

    TT_INLINE constexpr TT_HD IndexType& operator[](IndexType dim)
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

    TT_INLINE constexpr TT_HD IndexType operator[](IndexType dim) const
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

    TT_INLINE constexpr TT_HD void zero_()
    {
#pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
            indices[i] = 0;
        }
    }
};


// CUDA kernel argument that defines tensor layout
template <typename T, typename _IndexType, bool TIS_CUDA, int MAX_DIMS = -1>
struct TensorInfoBase
{
    static constexpr bool is_dynamic = MAX_DIMS == -1;
    static constexpr int max_dims    = is_dynamic ? MAX_TENSORINFO_DIMS : MAX_DIMS;
    using IndexType                  = _IndexType;
    using DimIndex                   = DimIndexStruct<max_dims, IndexType>;

    TensorInfoBase()
    {
        data  = nullptr;
        dims_ = 0;
    }
    TensorInfoBase(Tensor t)
    {
        if (!t.defined())
        {
            data = nullptr;
            return;
        }
        data  = t.template data_ptr<T>();
        dims_ = (int)t.dim();
        CHECK_LE(t.dim(), MAX_TENSORINFO_DIMS);
        for (int i = 0; i < max_dims; ++i)
        {
            if (i < t.dim())
            {
                // check for overflow, if 32bit indexing is used
                CHECK_LE(t.size(i), std::numeric_limits<IndexType>::max());
                CHECK_LE(t.stride(i), std::numeric_limits<IndexType>::max());
                sizes[i]   = (IndexType)t.size(i);
                strides[i] = (IndexType)t.stride(i);
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

    TT_INLINE constexpr TT_HD int dim()
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
    TT_INLINE constexpr TT_HD IndexType size(int64_t index) { return sizes[index]; }

    TT_INLINE constexpr TT_HD IndexType numel()
    {
        IndexType result = 1;
#pragma unroll
        for (int i = 0; i < max_dims; ++i)
        {
            if (i < dim())
            {
                result *= sizes[i];
            }
        }
        return result;
    }


    TT_INLINE constexpr TT_HD T& operator[](IndexType linearId)
    {
        if (contiguous)
        {
            return data[linearId];
        }
        return operator[](LinearIndexToDimIndex(linearId));
    }
    TT_INLINE constexpr TT_HD T& operator[](DimIndex index) { return data[IndexToOffset(index)]; }


    template <typename... Ts>
    TT_INLINE constexpr TT_HD T& operator()(Ts... args)
    {
        return operator[](IndexToOffset(DimIndex({args...})));
    }

    TT_INLINE constexpr TT_HD IndexType IndexToOffset(DimIndex index)
    {
        IndexType offset = 0;

#pragma unroll
        for (int i = 0; i < max_dims; ++i)
        {
            if (i < dim())
            {
#if TT_DEBUG
#    if defined(__CUDACC__)
#        if defined(TT_DEVICE_CODE)
                CUDA_KERNEL_ASSERT(index[i] >= 0);
                CUDA_KERNEL_ASSERT(index[i] < sizes[i]);
#        endif
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

    TT_INLINE constexpr TT_HD DimIndex LinearIndexToDimIndex(IndexType linearId)
    {
        DimIndex result;

#pragma unroll
        for (int j = 0; j < max_dims - 1; ++j)
        {
            auto i = max_dims - j - 1;
            // if(i==dim()) break;
            if (i < dim())
            {
                IndexType curDimIndex = linearId % sizes[i];
                result[i]             = curDimIndex;
                linearId /= sizes[i];
            }
        }

        result[0] = linearId;
        return result;
    }

    TT_INLINE constexpr TT_HD DimIndex LinearIndexToDimIndexSkipDim(IndexType linearId, IndexType dim_to_skip)
    {
        DimIndex result;

#pragma unroll
        for (int j = 0; j < max_dims - 1; ++j)
        {
            auto i = max_dims - j - 1;
            if (i < dim())
            {
                IndexType curDimIndex = linearId % sizes[i];
                IndexType siz         = sizes[i];

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
    TT_INLINE constexpr TT_HD bool index_in_range(DimIndex index)
    {
        bool result = true;
#pragma unroll
        for (int i = 0; i < max_dims; ++i)
        {
            if (i < dim() && (index[i] < 0 || index[i] > sizes[i]))
            {
                result = false;
            }
        }
        return result;
    }

    TT_INLINE constexpr TT_HD DimIndex clamp_index_to_size(DimIndex index)
    {
        DimIndex result;

#pragma unroll
        for (int i = 0; i < max_dims; ++i)
        {
            if (i == dim()) break;


            result[i] = std::clamp(IndexType(index[i]), IndexType(0), IndexType(sizes[i] - 1));
        }
        return result;
    }


    T* data = nullptr;
    IndexType sizes[max_dims];
    IndexType strides[max_dims];
    int dims_       = 0;
    bool contiguous = false;
};


// on the CPU always use 64-bit indexing
template <typename T, int MAX_DIMS = -1>
using TensorInfo = TensorInfoBase<T, int64_t, false, MAX_DIMS>;


template <typename T, int MAX_DIMS = -1>
using TensorInfoCuda = TensorInfoBase<T, int64_t, true, MAX_DIMS>;


}  // namespace tinytorch
