#pragma once

#include "torch/core/half.h"

#include "torch/tiny_torch_config.h"

namespace tinytorch
{
namespace UnaryOperators
{
struct Abs
{
    template <typename T>
    __forceinline__ TT_HD T forward(T v)
    {
        return v > T(0) ? v : -v;
    }

    template <typename T>
    __forceinline__ TT_HD T backward(T v)
    {
        return v < 0 ? -1 : v > 0 ? 1 : 0;
    }
};
struct Round
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return ::round(v);
    }
};
struct Sqrt
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return ::sqrt(v);
    }
    template <typename T>
    TT_HD T backward(T input_x, T grad_output)
    {
        float J = 0;
        if (input_x > T(0))
        {
            J = 1.f / (2.f * ::sqrt(float(input_x)));
        }
        return T(J) * grad_output;
    }
};
struct Log
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return ::log(v);
    }
};
struct Exp
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return ::exp(v);
    }
};
struct Sign
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return (v < T(0)) ? T(-1) : (v > T(0)) ? T(1) : T(0);
    }
};
struct Sin
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return ::sin(v);
    }
};
struct Cos
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return ::cos(v);
    }
};
struct Relu
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return (v > T(0.f)) ? v : T(0.f);
    }
};
struct Sigmoid
{
    template <typename T>
    TT_HD T forward(T x)
    {
        return T(1.f) / (T(1.f) + ::exp(-x));
    }
    template <typename T>
    TT_HD T backward(T input_x, T grad_output)
    {
        float x = 1.0f / (1.0f + expf(-input_x));
        T J     = (T)(x * (1.0f - x));
        return J * grad_output;
    }
};
struct Softplus
{
    Softplus(float beta) : beta(beta) { threshold = 20.f / beta; }
    template <typename T>
    TT_HD T forward(T x)
    {
        if (x > T(threshold)) return x;
        return T(::log(::exp(x * T(beta)) + T(1.f)) / T(beta));
    }
    template <typename T>
    TT_HD T backward(T input_x, T grad_output)
    {
        if (input_x > T(threshold)) return T(1.f) * grad_output;
        T tmp = expf((float)input_x * beta);
        T J   = (tmp / (tmp + T(1.f)));
        return grad_output * J;
    }
    float beta;
    float threshold;
};
}  // namespace UnaryOperators

namespace BinaryOperators
{
struct Add
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a + b;
    }
};
struct Sub
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a - b;
    }
};
struct Mult
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a * b;
    }
};
struct Div
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a / b;
    }
};
struct Equal
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a == b;
    }
};
struct Greater
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a > b;
    }
};
struct Less
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a < b;
    }
};
struct Pow
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return ::pow(a, b);
    }
};
struct Min
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a < b ? a : b;
    }
};
struct Max
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        return a > b ? a : b;
    }
};
}  // namespace BinaryOperators

//
// template <typename T>
// inline TT_HD T relu(T x)
//{
//    return (x > T(0.f)) ? x : T(0.f);
//}
//
// template <typename T>
// inline TT_HD T sigmoid(T x)
//{
//    return T(1.f) / (T(1.f) + ::exp(-x));
//}
//
// template <typename T>
// inline TT_HD T softplus(T x, T beta)
//{
//    return T(::log(T(1.f) + ::exp(beta * x)) / beta);
//}
//
//
//
// template <typename T>
// inline TT_HD std::pair<T, T> div_backward(T a, T b)
//{
//    return {T(1.f) / b, -a / (b * b)};
//}
//
// template <typename T>
// inline TT_HD T log_backward(T x)
//{
//    return T(1.f) / x;
//}
//
// template <typename T>
// inline TT_HD T log1p_backward(T x)
//{
//    return T(1.f) / (x + T(1.f));
//}
//
// template <typename T>
// inline TT_HD T pow_backward(T x, T b)
//{
//    return b * ::pow(x, b - 1);
//}
//
// template <typename T>
// inline TT_HD T sin_backward(T x)
//{
//    return ::cos(x);
//}
//
// template <typename T>
// inline TT_HD T cos_backward(T x)
//{
//    return -::sin(x);
//}
//
// template <typename T>
// inline TT_HD T relu_backward(T x)
//{
//    return ((x < T(0.f)) ? T(0.f) : T(1.f));
//}
//
// template <typename T>
// inline TT_HD T sigmoid_backward(T x)
//{
//    T expnegx = T(::exp(-x));
//    return expnegx / ((T(1.f) + expnegx) * (T(1.f) + expnegx));
//}
//
// template <typename T>
// inline TT_HD T softplus_backward(T x, T beta)
//{
//    T e = T(::exp(beta * x));
//    return e / (e + T(1.f));
//}

}  // namespace tinytorch
