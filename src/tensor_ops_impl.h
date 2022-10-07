#pragma once

#include "tensor.h"
namespace tinytorch
{


inline Tensor square_impl(Tensor a)
{
    Tensor result(a.size());
    for (int i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] * a[i];
    }
    return result;
}



inline Tensor sub_impl(Tensor a, Tensor b)
{
    Tensor result(a.size());
    for (int i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] - b[i];
    }
    return result;
}


inline Tensor add_impl(Tensor a, Tensor b)
{
    Tensor result(a.size());
    for (int i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] + b[i];
    }
    return result;
}

inline Tensor mult_impl(Tensor a, Tensor b)
{
    Tensor result(a.size());
    for (int i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] * b[i];
    }
    return result;
}


inline Tensor sum_impl(Tensor a)
{
    Tensor result = zero(1);
    for (int i = 0; i < a.size(); ++i)
    {
        result[0] += a[i];
    }
    return result;
}

// ================================================================================================================

inline std::vector<Tensor> square_backward_impl(Tensor a, Tensor grad_output)
{
    Tensor result(a.size());
    for (int i = 0; i < a.size(); ++i)
    {
        result[i] = 2 * a[i] * grad_output[i];
    }
    return {result};
}
inline std::vector<Tensor> mult_backward_impl(Tensor a, Tensor b, Tensor grad_output)
{
    Tensor result_a(a.size());
    Tensor result_b(a.size());
    for (int i = 0; i < a.size(); ++i)
    {
        result_a[i] = b[i] * grad_output[i];
        result_b[i] = a[i] * grad_output[i];
    }
    return {result_a, result_b};
}

inline std::vector<Tensor>  add_backward_impl(Tensor grad_output)
{
    Tensor result_a(grad_output.size());
    Tensor result_b(grad_output.size());
    for (int i = 0; i < grad_output.size(); ++i)
    {
        result_a[i] = grad_output[i];
        result_b[i] = grad_output[i];
    }
    return {result_a, result_b};
}

inline std::vector<Tensor>  sub_backward_impl(Tensor grad_output)
{
    Tensor result_a(grad_output.size());
    Tensor result_b(grad_output.size());
    for (int i = 0; i < grad_output.size(); ++i)
    {
        result_a[i] = grad_output[i];
        result_b[i] = -grad_output[i];
    }
    return {result_a, result_b};
}
inline std::vector<Tensor>  sum_backward_impl(int input_size, Tensor grad_output)
{
    assert(grad_output.size() == 1);
    Tensor result(input_size);
    for (int i = 0; i < input_size; ++i)
    {
        result[i] = grad_output[0];
    }
    return {result};
}


}  // namespace tinytorch