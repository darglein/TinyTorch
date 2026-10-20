/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <functional>

#include "test_utils.h"

#ifdef __linux__
#    include <sys/prctl.h>
#endif

namespace
{
using namespace tinytorch;

// Returns a small 2x3 tensor of known values on the given device.
Tensor base_tensor(Device d)
{
    return tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, d);
}

// Called in the child of a death test: a piped core_pattern (e.g. apport) runs the core dump handler even with
// ulimit -c 0, which adds about a second to every expected crash.
void DisableCoreDumps()
{
#ifdef __linux__
    prctl(PR_SET_DUMPABLE, 0);
#endif
}

const std::vector<ScalarType> kAllTypes   = {kUInt8, kUInt16, kInt16, kInt32, kLong, kHalf, kFloat, kDouble};
const std::vector<ScalarType> kFloatTypes = {kHalf, kFloat, kDouble};

bool is_float_type(ScalarType t)
{
    return t == kHalf || t == kFloat || t == kDouble;
}

// Random CPU tensor of dtype t with values in [lo, hi] (integers: [max(lo, 0 for unsigned), hi]).
Tensor rand_cpu(const SizeType& sizes, ScalarType t, double lo = -3, double hi = 3)
{
    if (is_float_type(t))
    {
        Tensor r = empty(sizes, TensorOptions().dtype(kFloat));
        uniform(r, lo, hi);
        return r.to(t);
    }
    int64_t l = (t == kUInt8 || t == kUInt16) ? 0 : (int64_t)lo;
    Tensor r  = empty(sizes, TensorOptions().dtype(kLong));
    uniform_int(r, l, (int64_t)hi + 1);
    return r.to(t);
}

// The same random tensor on the CPU and on the GPU, both viewed through perm (non-contiguous inputs for the kernels).
struct DevicePair
{
    Tensor cpu, cuda;
};
DevicePair rand_pair(const SizeType& sizes, ScalarType t, const SizeType& perm, double lo = -3, double hi = 3)
{
    Tensor base = rand_cpu(sizes, t, lo, hi);
    return {base.permute(perm), base.cuda().permute(perm)};
}

// Like tttest::tensor_near, but NaN == NaN and the elements are compared in logical order (to_double_vec reads a
// non-contiguous double tensor in storage order).
bool same(const Tensor& cpu, const Tensor& cuda, double rtol = 1e-4, double atol = 1e-4)
{
    Tensor a = cuda.cpu().contiguous();
    Tensor b = cpu.contiguous();
    if (!(a.sizes() == b.sizes()))
    {
        return false;
    }
    auto va = tttest::to_double_vec(a);
    auto vb = tttest::to_double_vec(b);
    for (size_t i = 0; i < va.size(); ++i)
    {
        if (std::isnan(va[i]) && std::isnan(vb[i]))
        {
            continue;
        }
        if (!std::isfinite(va[i]) || !std::isfinite(vb[i]))
        {
            if (va[i] != vb[i])
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

#define EXPECT_SAME(cpu_result, cuda_result, ...)                                                      \
    EXPECT_TRUE(same(cpu_result, cuda_result, ##__VA_ARGS__)) << "cpu:\n"                              \
                                                              << tttest::tp(cpu_result) << "\ncuda:\n" \
                                                              << tttest::tp((cuda_result).cpu())

// min(dim) and max(dim): values and indices
void expect_min_max_dim_same(const Tensor& a, const Tensor& ac, int64_t dim, bool keepdim)
{
    for (bool is_max : {false, true})
    {
        auto r  = is_max ? a.max(dim, keepdim) : a.min(dim, keepdim);
        auto rc = is_max ? ac.max(dim, keepdim) : ac.min(dim, keepdim);
        EXPECT_SAME(r.first, rc.first) << "max " << is_max;
        EXPECT_SAME(r.second, rc.second) << "max " << is_max;
    }
}

// Runs f on leaf copies of the inputs on device and backpropagates sum(f(inputs) * grad_out). Returns the result
// followed by the input gradients. An undefined grad_out is set to a random tensor of the result's shape.
std::vector<Tensor> forward_backward(Device device, const std::vector<Tensor>& inputs, Tensor& grad_out,
                                     const std::function<Tensor(const std::vector<Tensor>&)>& f)
{
    std::vector<Tensor> leaves;
    for (const auto& x : inputs)
    {
        leaves.push_back(x.clone().to(device));
        leaves.back().set_requires_grad(true, true);
    }
    Tensor r = f(leaves);
    if (!grad_out.defined())
    {
        grad_out = rand_cpu(r.sizes(), r.scalar_type());
    }
    tinytorch::backward((r * grad_out.to(device)).sum());
    std::vector<Tensor> res = {r.detach()};
    for (const auto& l : leaves)
    {
        res.push_back(l.grad());
    }
    return res;
}

void expect_forward_backward_same(const std::vector<Tensor>& inputs,
                                  const std::function<Tensor(const std::vector<Tensor>&)>& f)
{
    Tensor grad_out;
    auto cpu  = forward_backward(kCPU, inputs, grad_out, f);
    auto cuda = forward_backward(kCUDA, inputs, grad_out, f);
    EXPECT_SAME(cpu[0], cuda[0], 1e-3, 1e-3) << "result";
    for (size_t i = 1; i < cpu.size(); ++i)
    {
        EXPECT_SAME(cpu[i], cuda[i], 1e-3, 1e-3) << "grad of input " << i - 1;
    }
}

}  // namespace

// ------------------------------------------------------------------ device transfer

TEST(TinyTorchDevice, ToDeviceRoundTrip)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);

    Tensor c = a.to(kCUDA);
    EXPECT_EQ(c.device(), kCUDA);
    TT_EXPECT_CLOSE(c, a, 0, 0);

    Tensor back = c.to(kCPU);
    EXPECT_EQ(back.device(), kCPU);
    TT_EXPECT_CLOSE(back, a, 0, 0);
}

TEST(TinyTorchDevice, CudaCpuMethods)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);
    Tensor c = a.cuda();
    EXPECT_TRUE(c.is_cuda());
    EXPECT_FALSE(c.is_cpu());

    Tensor b = c.cpu();
    EXPECT_TRUE(b.is_cpu());
    TT_EXPECT_CLOSE(b, a, 0, 0);
}

TEST(TinyTorchDevice, ToInPlace)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);
    a.to_(kCUDA);
    EXPECT_EQ(a.device(), kCUDA);
    TT_EXPECT_CLOSE(a, base_tensor(kCPU), 0, 0);
    a.to_(kCPU);
    EXPECT_EQ(a.device(), kCPU);
}

TEST(TinyTorchDevice, CopyCrossDevice)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU);
    Tensor c = tinytorch::empty({2, 3}, TensorOptions().device(kCUDA));

    tinytorch::copy(a, c, false);
    TT_EXPECT_CLOSE(c, a, 0, 0);

    tinytorch::copy(c, a, false);
    TT_EXPECT_CLOSE(a, base_tensor(kCPU), 0, 0);
}

// A copy into a tensor with fewer elements fails the CHECK, also on the same-dtype contiguous memcpy path of the GPU.
TEST(TinyTorchDeviceDeathTest, CopyNumelMismatchOnCuda)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(
        {
            DisableCoreDumps();
            Tensor src = zeros({1000}, TensorOptions().device(kCUDA));
            Tensor dst = zeros({10}, TensorOptions().device(kCUDA));
            tinytorch::copy(src, dst);
        },
        "");
}

TEST(TinyTorchDevice, DtypeConversionCrossDevice)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = base_tensor(kCPU).to(kFloat64);
    Tensor c = a.to(kCUDA);

    EXPECT_EQ(c.device(), kCUDA);
    EXPECT_EQ(c.dtype(), kFloat64);
    TT_EXPECT_CLOSE(c, a, 1e-6, 1e-6);
}

TEST(TinyTorchDevice, DeviceHash)
{
    std::hash<Device> h;
    EXPECT_EQ(h(Device(kCUDA, 1)), h(Device(kCUDA, 1)));
    EXPECT_NE(h(Device(kCUDA, 0)), h(Device(kCUDA, 1)));
    EXPECT_NE(h(Device(kCPU)), h(Device(kCUDA, 0)));
}

// ------------------------------------------------------------------ autograd across devices

TEST(TinyTorchDevice, AutogradOnCuda)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor a = tttest::leaf({1, 2, 3, 4}, {2, 2}, kCUDA);
    Tensor b = tttest::leaf({10, 20, 30, 40}, {2, 2}, kCUDA);

    Tensor r = a * b;
    tinytorch::backward(tinytorch::sum(r));

    TT_EXPECT_CLOSE(a.grad(), b, 0, 0);
    TT_EXPECT_CLOSE(b.grad(), a, 0, 0);
}

TEST(TinyTorchDevice, AutogradWithDeviceMove)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    // move a leaf to the GPU, then build the graph there
    Tensor a = tttest::leaf({1, 2, 3, 4}, {4}, kCPU);
    a.to_(kCUDA);

    Tensor r = a * a;
    tinytorch::backward(tinytorch::sum(r));
    // d(a^2)/da = 2a
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({2, 4, 6, 8}, {4}, kCUDA), 0, 0);
}

// The gradient flowing back through to(device) may be non-contiguous (views created by cat/permute backward)
TEST(TinyTorchDevice, ToDeviceBackwardWithNonContiguousGrad)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    {
        Tensor x = tttest::leaf({1, 2, 3, 4, 5, 6}, {2, 3}, kCPU);
        Tensor w = tttest::leaf({0, 0}, {2, 1}, kCUDA);
        Tensor c = tttest::make_tensor({1, 2, 3, 4, 5, 6, 7, 8}, {2, 4}, kCUDA);
        Tensor y = x.to(kCUDA);
        tinytorch::backward(tinytorch::sum(tinytorch::cat({y, w}, 1) * c));
        EXPECT_EQ(x.grad().device(), kCPU);
        TT_EXPECT_CLOSE(x.grad(), tttest::make_tensor({1, 2, 3, 5, 6, 7}, {2, 3}, kCPU), 0, 0);
    }
    {
        Tensor x = tttest::leaf({1, 2, 3, 4, 5, 6}, {2, 3}, kCPU);
        Tensor c = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {3, 2}, kCUDA);
        Tensor y = x.to(kCUDA);
        tinytorch::backward(tinytorch::sum(y.permute({1, 0}) * c));
        TT_EXPECT_CLOSE(x.grad(), tttest::make_tensor({1, 3, 5, 2, 4, 6}, {2, 3}, kCPU), 0, 0);
    }
    {
        // and the other direction
        Tensor x = tttest::leaf({1, 2, 3, 4, 5, 6}, {2, 3}, kCUDA);
        Tensor c = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {3, 2}, kCPU);
        Tensor y = x.to(kCPU);
        tinytorch::backward(tinytorch::sum(y.permute({1, 0}) * c));
        TT_EXPECT_CLOSE(x.grad(), tttest::make_tensor({1, 3, 5, 2, 4, 6}, {2, 3}, kCUDA), 0, 0);
    }
}

// ------------------------------------------------------------------ CPU/CUDA agreement

// Runs the same computation on both devices and compares the results. Most inputs are random, odd-sized and
// non-contiguous (permuted views) to exercise the stride handling of the CUDA kernels.
class TinyTorchDeviceAgreement : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        if (!tttest::has_cuda())
        {
            GTEST_SKIP() << "no CUDA device available";
        }
    }

    NoGradGuard no_grad_;
};

TEST_F(TinyTorchDeviceAgreement, ElementwiseOpsMatch)
{
    Tensor a_cpu = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, kCPU);
    Tensor b_cpu = tttest::make_tensor({5, 6, 7, 8}, {2, 2}, kCPU);

    Tensor a_g = a_cpu.to(kCUDA);
    Tensor b_g = b_cpu.to(kCUDA);

    EXPECT_SAME(a_cpu + b_cpu, a_g + b_g);
    EXPECT_SAME(a_cpu - b_cpu, a_g - b_g);
    EXPECT_SAME(a_cpu * b_cpu, a_g * b_g);
    EXPECT_SAME(a_cpu / b_cpu, a_g / b_g);
    EXPECT_SAME(tinytorch::sqrt(a_cpu), tinytorch::sqrt(a_g));
    EXPECT_SAME(tinytorch::exp(a_cpu), tinytorch::exp(a_g));
    EXPECT_SAME(tinytorch::relu(a_cpu - 3.0), tinytorch::relu(a_g - 3.0));
    EXPECT_SAME(tinytorch::sigmoid(a_cpu), tinytorch::sigmoid(a_g));
    EXPECT_SAME(tinytorch::sum(a_cpu), tinytorch::sum(a_g));
    EXPECT_SAME(tinytorch::mean(a_cpu), tinytorch::mean(a_g));
    EXPECT_SAME(tinytorch::cumsum(a_cpu, 1), tinytorch::cumsum(a_g, 1));
    EXPECT_SAME(tinytorch::matmul(a_cpu, b_cpu), tinytorch::matmul(a_g, b_g));
    EXPECT_SAME(tinytorch::permute(a_cpu, {1, 0}), tinytorch::permute(a_g, {1, 0}));
    EXPECT_SAME(tinytorch::slice(a_cpu, 0, 1, 2, 1), tinytorch::slice(a_g, 0, 1, 2, 1));
}

TEST_F(TinyTorchDeviceAgreement, BinaryOpsNonContiguousBroadcast)
{
    for (auto t : kAllTypes)
    {
        SCOPED_TRACE(t);
        double tol = t == kHalf ? 1e-2 : 1e-5;
        // [7, 5, 9] op [7, 1, 9] (broadcast and permuted)
        auto [a, ac] = rand_pair({7, 5, 9}, t, {0, 1, 2}, 1, 5);
        auto [b, bc] = rand_pair({9, 1, 7}, t, {2, 1, 0}, 1, 5);
        EXPECT_SAME(a + b, ac + bc, tol, tol);
        EXPECT_SAME(a - b, ac - bc, tol, tol);
        EXPECT_SAME(a * b, ac * bc, tol, tol);
        if (is_float_type(t))
        {
            EXPECT_SAME(a / b, ac / bc, tol, tol);
            EXPECT_SAME(2.0 / b, 2.0 / bc, tol, tol);
        }
        EXPECT_SAME(tinytorch::min(a, b), tinytorch::min(ac, bc));
        EXPECT_SAME(tinytorch::max(a, b), tinytorch::max(ac, bc));
        EXPECT_SAME(a * 2.5, ac * 2.5, tol, tol);
        EXPECT_SAME(a + 1.5, ac + 1.5, tol, tol);
        EXPECT_SAME(b - 1.5, bc - 1.5, tol, tol);
        EXPECT_SAME(b.pow(2.0), bc.pow(2.0), tol, tol);
    }
}

TEST_F(TinyTorchDeviceAgreement, PowComparisonsInPlaceOpsAndMoments)
{
    for (auto t : kAllTypes)
    {
        SCOPED_TRACE(t);
        double tol   = t == kHalf ? 1e-2 : 1e-5;
        auto [a, ac] = rand_pair({4, 6, 5}, t, {2, 0, 1}, 0, 3);  // [5, 4, 6]
        auto [b, bc] = rand_pair({5, 1, 4}, t, {0, 2, 1}, 0, 3);  // [5, 4, 1]
        EXPECT_SAME(a.pow(b), ac.pow(bc), tol, tol);
        EXPECT_SAME(a == 2.0, ac == 2.0);
        EXPECT_SAME(a < 1.5, ac < 1.5);
        EXPECT_SAME(a > 1.5, ac > 1.5);

        Tensor x = a.clone(), xc = ac.clone();
        x += b;
        xc += bc;
        EXPECT_SAME(x, xc, tol, tol);
        x *= b;
        xc *= bc;
        EXPECT_SAME(x, xc, tol, tol);
        x -= 1.0;
        xc -= 1.0;
        EXPECT_SAME(x, xc, tol, tol);

        // in-place into a non-contiguous view
        auto [y, yc] = rand_pair({6, 4, 5}, t, {2, 1, 0}, 0, 3);
        y += b;
        yc += bc;
        EXPECT_SAME(y, yc, tol, tol);

        if (is_float_type(t))
        {
            EXPECT_SAME(a.std(), ac.std(), 3 * tol, 3 * tol);
            EXPECT_SAME(a.mean(), ac.mean(), 3 * tol, 3 * tol);
            EXPECT_SAME(a.mean(1, false), ac.mean(1, false), 3 * tol, 3 * tol);
            EXPECT_SAME(tinytorch::prod_sum(a), tinytorch::prod_sum(ac), 3 * tol, 3 * tol);
        }
    }
}

TEST_F(TinyTorchDeviceAgreement, UnaryOpsNonContiguous)
{
    for (auto t : kFloatTypes)
    {
        SCOPED_TRACE(t);
        double tol     = t == kHalf ? 1e-2 : 1e-5;
        auto [a, ac]   = rand_pair({6, 11, 3}, t, {2, 0, 1}, -2, 2);
        auto [ap, apc] = rand_pair({6, 11, 3}, t, {1, 2, 0}, 0.1, 2);  // positive, for log and sqrt
        EXPECT_SAME(a.abs(), ac.abs(), tol, tol);
        EXPECT_SAME(tinytorch::exp(a), tinytorch::exp(ac), tol, tol);
        EXPECT_SAME(tinytorch::sin(a), tinytorch::sin(ac), tol, tol);
        EXPECT_SAME(tinytorch::sigmoid(a), tinytorch::sigmoid(ac), tol, tol);
        EXPECT_SAME(tinytorch::relu(a), tinytorch::relu(ac), tol, tol);
        EXPECT_SAME(tinytorch::sign(a), tinytorch::sign(ac), tol, tol);
        EXPECT_SAME(tinytorch::softplus(a, 2.0), tinytorch::softplus(ac, 2.0), tol, tol);
        EXPECT_SAME(tinytorch::log(ap), tinytorch::log(apc), tol, tol);
        EXPECT_SAME(ap.sqrt(), apc.sqrt(), tol, tol);
        EXPECT_SAME(a.clamp(-0.5, 0.7), ac.clamp(-0.5, 0.7), tol, tol);
    }
}

// The GPU used to saturate negative floats to 0, the CPU wraps them (through int)
TEST_F(TinyTorchDeviceAgreement, ConvertNegativeFloatToUInt16)
{
    for (auto t : kFloatTypes)
    {
        Tensor a = tttest::make_tensor({-1, -2, 3, 0}, {4}, kCPU).to(t);
        EXPECT_SAME(a.to(kUInt16), a.cuda().to(kUInt16)) << t;
    }
}

TEST_F(TinyTorchDeviceAgreement, ReductionsNonContiguous)
{
    for (auto t : kAllTypes)
    {
        for (auto sizes : std::vector<SizeType>{{3, 5, 7}, {2, 70, 9}, {300, 3, 2}, {1, 1, 1}, {4, 1, 600}})
        {
            SCOPED_TRACE(::testing::Message() << t << " " << sizes);
            double tol   = t == kHalf ? 3e-2 : 1e-4;
            auto [a, ac] = rand_pair(sizes, t, {2, 0, 1});
            // values in [-1, 1] keep the products small
            auto [p, pc] = rand_pair(sizes, t, {2, 0, 1}, -1, 1);
            EXPECT_SAME(a.sum(), ac.sum(), tol, tol);
            EXPECT_SAME(a.min(), ac.min());
            EXPECT_SAME(a.max(), ac.max());
            EXPECT_SAME(tinytorch::abs_sum(a), tinytorch::abs_sum(ac), tol, tol);
            for (int d = 0; d < 3; ++d)
            {
                SCOPED_TRACE(d);
                EXPECT_SAME(a.sum(d, false), ac.sum(d, false), tol, tol);
                EXPECT_SAME(a.sum(d, true), ac.sum(d, true), tol, tol);
                expect_min_max_dim_same(a, ac, d, false);
                expect_min_max_dim_same(a, ac, d, true);
                EXPECT_SAME(a.cumsum(d), ac.cumsum(d), tol, tol);
                if (t != kHalf && sizes[0] < 100 && sizes[2] < 100)
                {
                    EXPECT_SAME(p.prod(d, false), pc.prod(d, false), tol, tol);
                    EXPECT_SAME(p.cumprod(d), pc.cumprod(d), tol, tol);
                }
            }
        }
    }
}

// More elements than the 1M threads of the global reduction grid, more dim reductions than its 4096 blocks and
// scans over several 256 element steps
TEST_F(TinyTorchDeviceAgreement, LargeReductionsAndScans)
{
    for (auto t : {kFloat, kDouble, kInt32})
    {
        SCOPED_TRACE(t);
        Tensor g  = rand_cpu({3, 1000, 701}, t, -2, 2);
        Tensor gc = g.cuda();
        EXPECT_SAME(g.sum(), gc.sum(), 1e-3, 1e-1);
        EXPECT_SAME(g.min(), gc.min());
        EXPECT_SAME(g.max(), gc.max());
        Tensor gp  = g.permute({2, 1, 0});
        Tensor gpc = gc.permute({2, 1, 0});
        for (int d = 0; d < 3; ++d)
        {
            SCOPED_TRACE(d);
            EXPECT_SAME(gp.sum(d, false), gpc.sum(d, false), 1e-3, 1e-3);
            EXPECT_SAME(gp.cumsum(d), gpc.cumsum(d), 1e-3, 1e-3);
            expect_min_max_dim_same(gp, gpc, d, false);
        }
    }
}

// MAX_TENSORINFO_DIMS dims
TEST_F(TinyTorchDeviceAgreement, HighDimReductions)
{
    auto [a, ac] = rand_pair({2, 3, 1, 4, 2, 3, 2, 5}, kFloat, {7, 6, 5, 4, 3, 2, 1, 0});
    for (int d = 0; d < 8; ++d)
    {
        SCOPED_TRACE(d);
        EXPECT_SAME(a.sum(d, true), ac.sum(d, true));
        EXPECT_SAME(a.prod(d, true), ac.prod(d, true));
        EXPECT_SAME(a.cumsum(d), ac.cumsum(d));
        expect_min_max_dim_same(a, ac, d, true);
    }
}

// min/max propagate NaN, also in the dim reductions (values and indices)
TEST_F(TinyTorchDeviceAgreement, NaNMinMax)
{
    for (auto t : kFloatTypes)
    {
        SCOPED_TRACE(t);
        Tensor a                = rand_cpu({5, 37}, kFloat);
        a.data_ptr<float>()[7]  = NAN;
        a.data_ptr<float>()[80] = NAN;
        a                       = a.to(t);
        Tensor ac               = a.cuda();
        EXPECT_SAME(a.min(), ac.min());
        EXPECT_SAME(a.max(), ac.max());
        EXPECT_SAME(a.sum(), ac.sum(), 1e-2, 1e-2);
        for (int d = 0; d < 2; ++d)
        {
            SCOPED_TRACE(d);
            expect_min_max_dim_same(a, ac, d, false);
        }
    }
}

TEST_F(TinyTorchDeviceAgreement, EmptyTensors)
{
    for (auto t : {kFloat, kInt32})
    {
        for (auto sizes : std::vector<SizeType>{{0, 3}, {3, 0}})
        {
            SCOPED_TRACE(::testing::Message() << t << " " << sizes);
            Tensor a  = empty(sizes, TensorOptions().dtype(t));
            Tensor ac = a.cuda();
            EXPECT_SAME(a + a, ac + ac);
            EXPECT_SAME(a * 2.0, ac * 2.0);
            EXPECT_SAME(a.abs(), ac.abs());
            EXPECT_SAME(a.sum(), ac.sum());
            EXPECT_SAME(a.min(), ac.min());
            EXPECT_SAME(a.max(), ac.max());
            for (int d = 0; d < 2; ++d)
            {
                EXPECT_SAME(a.sum(d, false), ac.sum(d, false)) << d;
                EXPECT_SAME(a.prod(d, false), ac.prod(d, false)) << d;
                EXPECT_SAME(a.cumsum(d), ac.cumsum(d)) << d;
                EXPECT_SAME(a.cumprod(d), ac.cumprod(d)) << d;
            }
            EXPECT_SAME(a.to(kDouble), ac.to(kDouble));
            EXPECT_SAME(a.clone(), ac.clone());
            EXPECT_SAME(a.permute({1, 0}).contiguous(), ac.permute({1, 0}).contiguous());
        }
    }
}

TEST_F(TinyTorchDeviceAgreement, IndexingNonContiguous)
{
    for (auto t : kAllTypes)
    {
        SCOPED_TRACE(t);
        auto [a, ac] = rand_pair({5, 6, 7}, t, {1, 2, 0});  // [6, 7, 5]
        for (int d = 0; d < 3; ++d)
        {
            SCOPED_TRACE(d);
            for (auto it : {kInt32, kLong})
            {
                Tensor idx = rand_cpu({9}, it, 0, a.size(d) - 1);
                EXPECT_SAME(a.index_select(d, idx), ac.index_select(d, idx.cuda()));
                if (is_float_type(t))
                {
                    SizeType ds = a.sizes();
                    ds[d]       = 9;
                    Tensor data = rand_cpu(ds, t);
                    EXPECT_SAME(a.index_add(d, idx, data), ac.index_add(d, idx.cuda(), data.cuda()), 2e-2, 2e-2);
                }
            }
            SizeType is = a.sizes();
            is[d]       = 4;
            Tensor gi   = rand_cpu(is, kLong, 0, a.size(d) - 1);
            EXPECT_SAME(a.gather(d, gi), ac.gather(d, gi.cuda()));
        }
        EXPECT_SAME(tinytorch::transpose(a, 0, 2), tinytorch::transpose(ac, 0, 2));
        EXPECT_SAME(a.permute({2, 0, 1}).contiguous(), ac.permute({2, 0, 1}).contiguous());
        EXPECT_SAME(a.repeat({2, 1, 3}), ac.repeat({2, 1, 3}));
        EXPECT_SAME(a.repeat_interleave(3), ac.repeat_interleave(3));
        EXPECT_SAME(a.clone(), ac.clone());
        for (auto t2 : kAllTypes)
        {
            EXPECT_SAME(a.to(t2), ac.to(t2)) << " -> " << t2;
        }
    }
}

TEST_F(TinyTorchDeviceAgreement, IndexCopyNonContiguousValues)
{
    for (auto t : kAllTypes)
    {
        Tensor a             = rand_cpu({5, 6, 7}, t);
        Tensor ac            = a.cuda();
        Tensor idx           = tttest::make_tensor(std::vector<int64_t>{4, 0, 2}, {3}, kCPU);
        auto [value, valuec] = rand_pair({5, 7, 3}, t, {0, 2, 1});
        tinytorch::index_copy(a, 1, idx, value);
        tinytorch::index_copy(ac, 1, idx.cuda(), valuec);
        EXPECT_SAME(a, ac) << t;
    }
}

TEST_F(TinyTorchDeviceAgreement, FillNonContiguous)
{
    for (auto t : kAllTypes)
    {
        SCOPED_TRACE(t);
        // fill along dim 0: values has the shape of a with size 1 in dim 0 (numel == a.size(0))
        auto [a, ac]  = rand_pair({2, 3, 6}, t, {2, 0, 1});
        Tensor values = rand_cpu({1, 2, 3}, t);
        tinytorch::fill(a, values, 0);
        tinytorch::fill(ac, values.cuda(), 0);
        EXPECT_SAME(a, ac);

        auto [b, bc] = rand_pair({4, 5, 6}, t, {2, 0, 1});
        tinytorch::fill(b, 0.0);
        tinytorch::fill(bc, 0.0);
        EXPECT_SAME(b, bc);
        b.fill_(2);
        bc.fill_(2);
        EXPECT_SAME(b, bc);
    }
}

TEST_F(TinyTorchDeviceAgreement, PaddingMatmulConv)
{
    for (auto t : kFloatTypes)
    {
        SCOPED_TRACE(t);
        double tol   = t == kHalf ? 3e-2 : 1e-4;
        auto [x, xc] = rand_pair({2, 3, 7, 5}, t, {0, 1, 3, 2});
        auto [y, yc] = rand_pair({2, 3, 4, 5, 6}, t, {0, 1, 4, 2, 3});
        for (auto m : {kZero, kBorder, kReflect})
        {
            EXPECT_SAME(tinytorch::padding_2d(x, 1, 2, 3, 0, m), tinytorch::padding_2d(xc, 1, 2, 3, 0, m)) << m;
            EXPECT_SAME(tinytorch::padding_3d(y, 1, 2, 3, 0, 2, 1, m), tinytorch::padding_3d(yc, 1, 2, 3, 0, 2, 1, m))
                << m;
        }

        auto [ma, mac] = rand_pair({3, 1, 5, 4}, t, {0, 1, 3, 2});
        Tensor mb      = rand_cpu({2, 5, 6}, t);
        EXPECT_SAME(tinytorch::matmul(ma, mb), tinytorch::matmul(mac, mb.cuda()), tol, tol);

        // stride 2, padding 1, dilation 2 with a non-contiguous weight; depthwise (groups 3) without bias
        Tensor w  = rand_cpu({6, 3, 3, 2}, t).permute({0, 1, 3, 2}) * 0.3;
        Tensor bi = rand_cpu({6}, t);
        Tensor xi = rand_cpu({2, 3, 9, 8}, t);
        EXPECT_SAME(tinytorch::conv2d(xi, w, bi, 2, 1, 2, 1),
                    tinytorch::conv2d(xi.cuda(), w.cuda(), bi.cuda(), 2, 1, 2, 1), tol, tol);
        Tensor wg = rand_cpu({6, 1, 3, 3}, t);
        EXPECT_SAME(tinytorch::conv2d(xi, wg, {}, 1, 1, 1, 3), tinytorch::conv2d(xi.cuda(), wg.cuda(), {}, 1, 1, 1, 3),
                    tol, tol);
    }
}

TEST_F(TinyTorchDeviceAgreement, GridSampleAndConvBackward)
{
    AutoGradMode grad_mode(true);
    for (auto t : {kFloat, kDouble})
    {
        for (auto interp : {kBilinear, kNearest})
        {
            for (auto pad : {kZero, kBorder, kReflect})
            {
                for (bool align_corners : {false, true})
                {
                    SCOPED_TRACE(::testing::Message() << t << " " << interp << " " << pad << " " << align_corners);
                    nn::functional::GridSampleFuncOptions o;
                    o.it = interp;
                    o.pm = pad;
                    o.ac = align_corners;
                    expect_forward_backward_same({rand_cpu({2, 3, 6, 7}, t), rand_cpu({2, 4, 5, 2}, t, -1.3, 1.3)},
                                                 [&](const std::vector<Tensor>& in)
                                                 { return nn::functional::grid_sample(in[0], in[1], o); });
                }
            }
        }
        SCOPED_TRACE(t);
        // stride 2, padding 2, dilation 2, groups 2
        expect_forward_backward_same({rand_cpu({2, 4, 9, 8}, t), rand_cpu({6, 2, 3, 2}, t), rand_cpu({6}, t)},
                                     [](const std::vector<Tensor>& in)
                                     { return tinytorch::conv2d(in[0], in[1], in[2], 2, 2, 2, 2); });
    }
}

// ------------------------------------------------------------------ MultiDeviceTensor (CUDA only)

#ifdef TT_HAS_CUDA
TEST(TinyTorchMultiDevice, SingleDeviceBasics)
{
    if (!tttest::has_cuda())
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    Tensor main = base_tensor(kCUDA);
    tinytorch::cuda::MultiDeviceTensor mdt(main, std::vector<Device>{kCUDA});

    EXPECT_EQ(mdt.size(), 1);
    EXPECT_TRUE(mdt.Initialized());
    EXPECT_TRUE(mdt.defined());
    TT_EXPECT_CLOSE(mdt.Main(), main, 0, 0);
    TT_EXPECT_CLOSE(mdt[Device(kCUDA, 0)], main, 0, 0);

    mdt.zero_();
    TT_EXPECT_CLOSE(mdt.Main(), tinytorch::zeros({2, 3}, TensorOptions().device(kCUDA)), 0, 0);
}
#endif  // TT_HAS_CUDA
