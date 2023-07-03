#pragma once

#include "torch/tiny_torch_config.h"

namespace tinytorch
{
#ifdef TT_HAS_CUDA
#    include <cuda_runtime_api.h>
#    define TT_HD __host__ __device__
#else
#    define TT_HD
#endif


template <typename T>
inline TT_HD T relu(T x)
{
    return (x > T(0)) ? x : T(0);
}

template <typename T>
inline TT_HD T sigmoid(T x)
{
    return T(1) / (T(1) + ::exp(-x));
}

template <typename T>
inline TT_HD T softplus(T x, T beta)
{
    return T(::log(1 + ::exp(beta * x)) / beta);
}



template <typename T>
inline TT_HD std::pair<T, T> div_backward(T a, T b)
{
    return {T(1) / b, -a / (b * b)};
}

template <typename T>
inline TT_HD T log_backward(T x)
{
    return T(1) / x;
}

template <typename T>
inline TT_HD T log1p_backward(T x)
{
    return T(1) / (x + T(1));
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
    return ((x < T(0)) ? T(0) : T(1));
}

template <typename T>
inline TT_HD T sigmoid_backward(T x)
{
    T expnegx = T(::exp(-x));
    return expnegx / ((T(1) + expnegx) * (T(1) + expnegx));
}

template <typename T>
inline TT_HD T softplus_backward(T x, T beta)
{
    T e = T(::exp(beta * x));
    return e / (e + T(1));
}



}  // namespace tinytorch
