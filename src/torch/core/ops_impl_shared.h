#pragma once

#include "torch/tiny_torch_config.h"

namespace tinytorch
{
template <typename T>
inline TT_HD T relu(T x)
{
    return (x > T(0.f)) ? x : T(0.f);
}

template <typename T>
inline TT_HD T sigmoid(T x)
{
    return T(1.f) / (T(1.f) + ::exp(-x));
}

template <typename T>
inline TT_HD T softplus(T x, T beta)
{
    return T(::log(T(1.f) + ::exp(beta * x)) / beta);
}



template <typename T>
inline TT_HD std::pair<T, T> div_backward(T a, T b)
{
    return {T(1.f) / b, -a / (b * b)};
}

template <typename T>
inline TT_HD T log_backward(T x)
{
    return T(1.f) / x;
}

template <typename T>
inline TT_HD T log1p_backward(T x)
{
    return T(1.f) / (x + T(1.f));
}

template <typename T>
inline TT_HD T pow_backward(T x, T b)
{
    return b * ::pow(x, b - 1);
}

template <typename T>
inline TT_HD T sin_backward(T x)
{
    return ::cos(x);
}

template <typename T>
inline TT_HD T cos_backward(T x)
{
    return -::sin(x);
}

template <typename T>
inline TT_HD T relu_backward(T x)
{
    return ((x < T(0.f)) ? T(0.f) : T(1.f));
}

template <typename T>
inline TT_HD T sigmoid_backward(T x)
{
    T expnegx = T(::exp(-x));
    return expnegx / ((T(1.f) + expnegx) * (T(1.f) + expnegx));
}

template <typename T>
inline TT_HD T softplus_backward(T x, T beta)
{
    T e = T(::exp(beta * x));
    return e / (e + T(1.f));
}

inline TT_HD int64_t index_along_dim(int64_t linearId, int64_t dims, int64_t dim, int64_t* input_sizes,
                               int64_t* input_strides)
{
    int64_t input_offset = 0;
    for (int64_t i = dims - 1; i > 0; --i)
    {
        if (i != dim)
        {
            int64_t curDimIndex = linearId % input_sizes[i];
            input_offset += curDimIndex * input_strides[i];
            linearId /= input_sizes[i];
        }
    }

    if (dim != 0)
    {
        input_offset += linearId * input_strides[0];
    }

    return input_offset;
}

inline TT_HD void calculate_offsets(int64_t linearId, int64_t dims, int64_t* a_sizes, int64_t* b_sizes,
                                    int64_t* a_strides, int64_t* b_strides, int64_t& offset_a, int64_t& offset_b)
{
    // This handles the case that if one tensor has size 1 along a dimension, the respective value is duplicated along
    // this dimension.

    offset_a = 0;
    offset_b = 0;

    for (int64_t i = dims - 1; i > 0; --i)
    {
        int64_t sa    = a_sizes[i];
        int64_t sb    = b_sizes[i];
        int64_t max_s = (sa > sb) ? sa : sb;

        offset_a += (sa == 1) ? 0 : ((linearId % sa) * a_strides[i]);
        offset_b += (sb == 1) ? 0 : ((linearId % sb) * b_strides[i]);
        linearId /= max_s;
    }

    int64_t sa = a_sizes[0];
    int64_t sb = b_sizes[0];

    offset_a += (sa == 1) ? 0 : ((linearId % sa) * a_strides[0]);
    offset_b += (sb == 1) ? 0 : ((linearId % sb) * b_strides[0]);
}



}  // namespace tinytorch
