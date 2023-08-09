/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "torch/core/tensor.h"


namespace tinytorch
{

#define CASE_MACRO(func, scalar_type, ...) \
    case scalar_type:                      \
        func(__VA_ARGS__);                 \
        break;

#define SWITCH_MACRO_INT(real_scalar_type, func, ...)                  \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func<int32_t>, kInt32, __VA_ARGS__)                 \
        CASE_MACRO(func<int64_t>, kInt64, __VA_ARGS__)                 \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }

#define SWITCH_MACRO_FLOAT(real_scalar_type, func, ...)                \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func<Half>, kHalf, __VA_ARGS__)                     \
        CASE_MACRO(func<float>, kFloat, __VA_ARGS__)                   \
        CASE_MACRO(func<double>, kDouble, __VA_ARGS__)                 \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }

// TODO: Half!
#define SWITCH_MACRO_ALL(real_scalar_type, func, ...)                  \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func<uint8_t>, kUInt8, __VA_ARGS__)                 \
        CASE_MACRO(func<int16_t>, kInt16, __VA_ARGS__)                 \
        CASE_MACRO(func<int32_t>, kInt32, __VA_ARGS__)                 \
        CASE_MACRO(func<int64_t>, kLong, __VA_ARGS__)                  \
        CASE_MACRO(func<Half>, kHalf, __VA_ARGS__)                     \
        CASE_MACRO(func<float>, kFloat, __VA_ARGS__)                   \
        CASE_MACRO(func<double>, kDouble, __VA_ARGS__)                 \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }

// used for conversion
#define SWITCH_MACRO_ALL_DUAL(real_scalar_type, second_type, func, ...) \
    switch (real_scalar_type)                                           \
    {                                                                   \
        CASE_MACRO((func<uint8_t, second_type>), kUInt8, __VA_ARGS__)   \
        CASE_MACRO((func<int16_t, second_type>), kInt16, __VA_ARGS__)   \
        CASE_MACRO((func<int32_t, second_type>), kInt32, __VA_ARGS__)   \
        CASE_MACRO((func<int64_t, second_type>), kLong, __VA_ARGS__)    \
        CASE_MACRO((func<Half, second_type>), kHalf, __VA_ARGS__)       \
        CASE_MACRO((func<float, second_type>), kFloat, __VA_ARGS__)     \
        CASE_MACRO((func<double, second_type>), kDouble, __VA_ARGS__)   \
        default:                                                        \
            CHECK(false) << "invalid input type " << real_scalar_type;  \
    }


static SizeType max_size(Tensor a, Tensor b)
{
    CHECK_EQ(a.dim(), b.dim());
    SizeType new_sizes;
    new_sizes.resize(a.dim());
    for (int64_t i = 0; i < a.dim(); ++i)
    {
        int64_t as = a.size(i);
        int64_t bs = b.size(i);
        CHECK(as == bs || as == 1 || bs == 1);
        new_sizes[i] = std::max(as, bs);
    }
    return new_sizes;
}

}  // namespace tinytorch
