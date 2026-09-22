/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "torch/tiny_torch.h"

#ifdef TT_HAS_CUDA
#    include "cuda_runtime.h"
#endif

namespace tttest
{

using tinytorch::Device;
using tinytorch::ScalarType;
using tinytorch::SizeType;
using tinytorch::Tensor;
using tinytorch::TensorOptions;
using tinytorch::kCUDA;
using tinytorch::kCPU;
using tinytorch::kFloat;
using tinytorch::kFloat64;
using tinytorch::kLong;

// ------------------------------------------------------------------ devices

inline bool has_cuda()
{
#ifdef TT_HAS_CUDA
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

// CPU always; CUDA only when the library was built with CUDA support.
inline std::vector<Device> all_devices()
{
    std::vector<Device> d{kCPU};
#ifdef TT_HAS_CUDA
    d.push_back(kCUDA);
#endif
    return d;
}

// Device-parameterized fixture: every test using it runs once per device.
struct DeviceTest : public ::testing::TestWithParam<Device>
{
    Device device() const { return GetParam(); }
    bool on_cuda() const { return device() == kCUDA; }

    void SetUp() override
    {
#ifdef TT_HAS_CUDA
        if (on_cuda())
        {
            if (!has_cuda())
            {
                GTEST_SKIP() << "no CUDA device available";
            }
            cudaSetDevice(device().index());
        }
#endif
    }
};

inline std::string device_name(const testing::TestParamInfo<Device>& info)
{
    return info.param.type() == kCUDA ? "CUDA" : "CPU";
}

// Declares the parameterized fixture (if it does not already exist) and registers
// one run per device: CPU always, CUDA when the library was built with it.
#define TT_INSTANTIATE_DEVICE_TESTS(SuiteName)                                                     \
    class SuiteName : public tttest::DeviceTest                                                    \
    {                                                                                               \
    };                                                                                              \
    INSTANTIATE_TEST_SUITE_P(AllDevices, SuiteName, ::testing::ValuesIn(tttest::all_devices()),    \
                              [](const ::testing::TestParamInfo<tttest::Device>& info)             \
                              {                                                                     \
                                  return tttest::device_name(info);                                \
                              })

// ------------------------------------------------------------- construction

// Builds an owned tensor on `device` from `data` (flattened, row-major).
// from_blob does not take ownership, so immediately copy into an owned tensor.
inline Tensor make_tensor(const std::vector<float>& data, const SizeType& sizes, Device device,
                          ScalarType dtype = kFloat)
{
    Tensor src = tinytorch::from_blob(static_cast<void*>(const_cast<float*>(data.data())), sizes);
    // cross-device copy requires identical dtypes: convert on the source device first
    if (src.dtype() != dtype)
    {
        src = src.to(dtype);
    }
    Tensor dst = tinytorch::empty(sizes, TensorOptions().device(device).dtype(dtype));
    tinytorch::copy(src, dst, false);
    return dst;
}

inline Tensor make_tensor(const std::vector<int64_t>& data, const SizeType& sizes, Device device)
{
    Tensor src = tinytorch::from_blob(static_cast<void*>(const_cast<int64_t*>(data.data())), sizes, kLong);
    Tensor dst = tinytorch::empty(sizes, TensorOptions().device(device).dtype(kLong));
    tinytorch::copy(src, dst, false);
    return dst;
}

// Brace lists ({1, 2, ...}) are ambiguous between the vector overloads above;
// this initializer_list overload wins for them and always builds a float tensor.
// Use an explicit std::vector<int64_t> for integer data.
inline Tensor make_tensor(std::initializer_list<float> data, const SizeType& sizes, Device device,
                          ScalarType dtype = kFloat)
{
    return make_tensor(std::vector<float>(data), sizes, device, dtype);
}

// -------------------------------------------------------------- comparison

inline std::vector<double> to_double_vec(const Tensor& t)
{
    Tensor d      = t.cpu().to(kFloat64);
    const double* p = d.data_ptr<double>();
    return {p, p + d.numel()};
}

// |a - b| <= atol + rtol * |b|  element-wise (also catches shape mismatch).
inline bool tensor_near(const Tensor& a, const Tensor& b, double rtol = 1e-4, double atol = 1e-5)
{
    if (!a.defined() || !b.defined())
    {
        return a.defined() == b.defined();
    }
    if (!(a.sizes() == b.sizes()))
    {
        return false;
    }
    std::vector<double> va = to_double_vec(a);
    std::vector<double> vb = to_double_vec(b);
    for (size_t i = 0; i < va.size(); ++i)
    {
        if (!std::isfinite(va[i]) || !std::isfinite(vb[i]))
        {
            if (!(va[i] == vb[i]))
            {
                return false;
            }
            continue;
        }
        if (std::abs(va[i] - vb[i]) > atol + rtol * std::abs(vb[i]))
        {
            return false;
        }
    }
    return true;
}

// Streaming a Tensor aborts on undefined tensors (CHECK in operator<<); this wrapper
// prints a placeholder instead so a failed assertion cannot take down the whole run.
struct TensorPrint
{
    const Tensor& t;
};
inline std::ostream& operator<<(std::ostream& os, const TensorPrint& p)
{
    if (p.t.defined())
    {
        tinytorch::operator<<(os, p.t);
    }
    else
    {
        os << "(undefined tensor)";
    }
    return os;
}
inline TensorPrint tp(const Tensor& t)
{
    return {t};
}

#define TT_EXPECT_CLOSE(actual, expected, rtol, atol)                                             \
    EXPECT_TRUE(tttest::tensor_near(actual, expected, rtol, atol))                                \
        << "actual:\n" << tttest::tp(actual) << "\nexpected:\n" << tttest::tp(expected)

// ------------------------------------------------------------- leaf helper

// A leaf tensor (requires grad + allocated gradient) on the given device.
inline Tensor leaf(const std::vector<float>& data, const SizeType& sizes, Device device,
                   ScalarType dtype = kFloat)
{
    Tensor t = make_tensor(data, sizes, device, dtype);
    t.set_requires_grad(true, true);
    return t;
}

// --------------------------------------------------------------- autograd

// A scalar (numel == 1) loss built from the (leaf) parameter tensors.
using LossFn = std::function<Tensor(const std::vector<Tensor>&)>;

// Verifies that the analytic gradients of f(params) match central finite differences.
// Expects every param to be a leaf with requires_grad and an allocated gradient.
// `params` is passed by value: the elements are mutable copies of the same handles.
inline void check_grads(std::vector<Tensor> params, const LossFn& f, double rtol = 2e-2,
                        double atol = 2e-3, double h = 1e-2)
{
    // analytic pass
    for (auto& p : params)
    {
        if (p.grad().defined())
        {
            p.mutable_grad().zero_();
        }
    }

    Tensor loss = f(params);
    tinytorch::backward(loss);

    // numeric pass: perturb one element at a time via fresh flat copies (view ops
    // abort on requires-grad tensors with grad mode on, so no in-place/view is used)
    std::vector<double> g_ana_vec;
    for (size_t pi = 0; pi < params.size(); ++pi)
    {
        const Tensor& p    = params[pi];
        std::vector<double> base = to_double_vec(p);
        g_ana_vec              = to_double_vec(p.grad());

        for (size_t i = 0; i < base.size(); ++i)
        {
            std::vector<double> plus = base;
            plus[i] += h;
            std::vector<double> minus = base;
            minus[i] -= h;

            std::vector<Tensor> pp = params;
            pp[pi] = leaf(std::vector<float>(plus.begin(), plus.end()), p.sizes(), p.device());
            double l_plus = f(pp).toFloat();

            std::vector<Tensor> pm = params;
            pm[pi] = leaf(std::vector<float>(minus.begin(), minus.end()), p.sizes(), p.device());
            double l_minus = f(pm).toFloat();

            double g_num = (l_plus - l_minus) / (2.0 * h);
            double g_ana = g_ana_vec[i];

            EXPECT_NEAR(g_ana, g_num, atol + rtol * std::abs(g_num))
                << "param " << pi << " element " << i << " (analytic " << g_ana << ", numeric " << g_num << ")";
        }
    }
}

}  // namespace tttest

// Make the common tinytorch names available unqualified in the test files.
using tinytorch::Device;
using tinytorch::ScalarType;
using tinytorch::SizeType;
using tinytorch::Tensor;
using tinytorch::TensorOptions;
using tinytorch::kCUDA;
using tinytorch::kCPU;
using tinytorch::kFloat;
using tinytorch::kFloat64;
using tinytorch::kInt32;
using tinytorch::kLong;
using tinytorch::kZero;
using tinytorch::kBorder;
using tinytorch::kReflect;
using tinytorch::kBilinear;
using tinytorch::kNearest;
