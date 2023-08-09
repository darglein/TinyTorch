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
    TT_HD T forward(T v)
    {
        return ::abs(v);
    }

    template <typename T>
    TT_HD T backward(T v)
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
};
struct Softplus
{
    Softplus(float beta) : beta(beta) {}
    template <typename T>
    TT_HD T forward(T x)
    {
        return T(::log(T(1.f) + ::exp(T(beta) * x)) / T(beta));
    }
    float beta;
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
        return ::pow(a,b);
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
