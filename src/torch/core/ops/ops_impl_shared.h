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



inline std::pair<SizeType, SizeType> CheckOperatorSizeMatchOneDim(const Tensor& a, const Tensor& b)
{
    CHECK_EQ(a.device(), b.device());
    CHECK_EQ(a.dim(), b.dim());

    std::vector<int64_t> expand_a, expand_b;
    for (int i = 0; i < a.dim(); ++i)
    {
        if (a.size(i) != b.size(i))
        {
            CHECK(a.size(i) == 1 || b.size(i) == 1) << "Size Missmatch " << a.sizes() << " " << b.sizes();

            // make sure we don't expand a 0 to a 1
            if (a.size(i) == 1 && b.size(i) > 1)
            {
                expand_a.push_back(i);
            }
            else if (b.size(i) == 1 && a.size(i) > 1)
            {
                expand_b.push_back(i);
            }
        }
    }
    return {expand_a, expand_b};
}

inline void BackwardExpand(Tensor& grad_a, Tensor& grad_b, SizeType expand_a, SizeType expand_b)
{
    if (grad_a.defined() && expand_a.size() > 0)
    {
        grad_a = grad_a.sum(expand_a, true);
    }
    if (grad_b.defined() && expand_b.size() > 0)
    {
        grad_b = grad_b.sum(expand_b, true);
    }
}

// Operators can have the case that one Tensor is dimension 1 along one axis and the other is not.
// This computes the size of the result tensor and checks if everything else is ok.
inline SizeType max_size(Tensor a, Tensor b)
{
    CHECK_EQ(a.dim(), b.dim());
    SizeType new_sizes;
    new_sizes.resize(a.dim());
    for (int64_t i = 0; i < a.dim(); ++i)
    {
        int64_t as = a.size(i);
        int64_t bs = b.size(i);
        CHECK(as == bs || as == 1 || bs == 1);
        if (as == 0 || bs == 0)
        {
            // 0-sized dims are not expanded
            new_sizes[i] = 0;
        }
        else
        {
            new_sizes[i] = std::max(as, bs);
        }
    }
    return new_sizes;
}


}  // namespace tinytorch
