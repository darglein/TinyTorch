#pragma once

#include "torch/core/half.h"

#include <cmath>
#include <limits>

#include "torch/tiny_torch_config.h"
#include <type_traits>

namespace tinytorch
{
// floating point type the math functions are evaluated in: double for double, float for everything else
template <typename T>
struct MathComputeType
{
    using Type = float;
};
template <>
struct MathComputeType<double>
{
    using Type = double;
};

// type in which an operation between a tensor element of type T and a double scalar is computed
template <typename T>
struct ScalarComputeType
{
    using Type =
        typename std::conditional<std::is_integral<T>::value, double, typename CpuComputeFloatType<T>::Type>::type;
};

// converts a result computed in ScalarComputeType<T> back to T (integers wrap like integer arithmetic)
template <typename T, typename S>
TT_HD inline T scalar_result_to(S v)
{
    if constexpr (std::is_integral<T>::value && !std::is_integral<S>::value)
    {
        return T(static_cast<long long>(v));
    }
    else
    {
        return T(v);
    }
}

// converts a finite clamp bound to T; integer bounds are rounded inwards and saturated to the range of T
template <typename T>
TT_HD inline T clamp_bound_to(double bound, bool is_low)
{
    if constexpr (std::is_integral<T>::value)
    {
        bound = is_low ? std::ceil(bound) : std::floor(bound);
        if (bound <= (double)std::numeric_limits<T>::lowest()) return std::numeric_limits<T>::lowest();
        if (bound >= (double)std::numeric_limits<T>::max()) return std::numeric_limits<T>::max();
    }
    return T(bound);
}

template <typename T>
TT_HD inline bool tt_isnan(T v)
{
    if constexpr (std::is_integral<T>::value)
    {
        return false;
    }
    else
    {
        return !(v == v);
    }
}

namespace UnaryOperators
{
struct Abs
{
    template <typename T>
    TT_HD T forward(T v)
    {
        return v > T(0) ? v : -v;
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
        using M = typename MathComputeType<T>::Type;
        return T(std::round(M(v)));
    }
};
struct Sqrt
{
    template <typename T>
    TT_HD T forward(T v)
    {
        using M = typename MathComputeType<T>::Type;
        return T(std::sqrt(M(v)));
    }
    template <typename T>
    TT_HD T backward(T input_x, T grad_output)
    {
        using M = typename MathComputeType<T>::Type;
        M J     = 0;
        if (input_x > T(0))
        {
            J = M(1) / (M(2) * std::sqrt(M(input_x)));
        }
        return T(J * M(grad_output));
    }
};
struct Log
{
    template <typename T>
    TT_HD T forward(T v)
    {
        using M = typename MathComputeType<T>::Type;
        return T(std::log(M(v)));
    }
};
struct Exp
{
    template <typename T>
    TT_HD T forward(T v)
    {
        using M = typename MathComputeType<T>::Type;
        return T(std::exp(M(v)));
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
        using M = typename MathComputeType<T>::Type;
        return T(std::sin(M(v)));
    }
};
struct Cos
{
    template <typename T>
    TT_HD T forward(T v)
    {
        using M = typename MathComputeType<T>::Type;
        return T(std::cos(M(v)));
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
        using M = typename MathComputeType<T>::Type;
        return T(M(1) / (M(1) + std::exp(-M(x))));
    }
    template <typename T>
    TT_HD T backward(T input_x, T grad_output)
    {
        using M = typename MathComputeType<T>::Type;
        M x     = M(1) / (M(1) + std::exp(-M(input_x)));
        return T(x * (M(1) - x) * M(grad_output));
    }
};
struct Softplus
{
    Softplus(double beta) : beta(beta) { threshold = 20.0 / beta; }
    template <typename T>
    TT_HD T forward(T x)
    {
        using M = typename MathComputeType<T>::Type;
        if (M(x) > M(threshold)) return x;
        return T(std::log(std::exp(M(x) * M(beta)) + M(1)) / M(beta));
    }
    template <typename T>
    TT_HD T backward(T input_x, T grad_output)
    {
        using M = typename MathComputeType<T>::Type;
        if (M(input_x) > M(threshold)) return grad_output;
        M tmp = std::exp(M(input_x) * M(beta));
        M J   = tmp / (tmp + M(1));
        return T(M(grad_output) * J);
    }
    double beta;
    double threshold;
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
        if (tt_isnan(a)) return a;
        if (tt_isnan(b)) return b;
        return b < a ? b : a;
    }
};
struct Max
{
    template <typename T>
    TT_HD T forward(T a, T b)
    {
        if (tt_isnan(a)) return a;
        if (tt_isnan(b)) return b;
        return a < b ? b : a;
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

            // a size-1 dim is broadcast to the other size (also to 0, like PyTorch)
            if (a.size(i) == 1)
            {
                expand_a.push_back(i);
            }
            else
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
        new_sizes[i] = (as == 1) ? bs : as;
    }
    return new_sizes;
}


}  // namespace tinytorch
