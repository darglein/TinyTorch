/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "torch/core/tensor.h"


namespace tinytorch
{

#define CASE_MACRO(func, type, scalar_type, ...) \
    case scalar_type:                            \
        func<type>(__VA_ARGS__);                 \
        break;

#define SWITCH_MACRO_FLOAT(real_scalar_type, func, ...)                \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, float, kFloat, __VA_ARGS__)                   \
        CASE_MACRO(func, double, kDouble, __VA_ARGS__)                 \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }

// TODO: Half!
#define SWITCH_MACRO_ALL(real_scalar_type, func, ...)                  \
    switch (real_scalar_type)                                          \
    {                                                                  \
        CASE_MACRO(func, uint8_t, kUInt8, __VA_ARGS__)                 \
        CASE_MACRO(func, int16_t, kInt16, __VA_ARGS__)                 \
        CASE_MACRO(func, int32_t, kInt32, __VA_ARGS__)                 \
        CASE_MACRO(func, int64_t, kLong, __VA_ARGS__)                  \
        CASE_MACRO(func, float, kFloat, __VA_ARGS__)                   \
        CASE_MACRO(func, double, kDouble, __VA_ARGS__)                 \
        default:                                                       \
            CHECK(false) << "invalid input type " << real_scalar_type; \
    }



}