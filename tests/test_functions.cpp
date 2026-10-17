/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"
#include "torch/core/tensor_info.h"

using tttest::DeviceTest;

namespace
{

// Folds in the cases from the former tests/test_backward.cpp.

// d(sum(A*B))/dA[p][q] = sum_n B[q][n] (row sums of B),
// d(sum(A*B))/dB[q][n] = sum_p A[p][q] (column sums of A, repeated over n)
TEST(TinyTorchBackward, Matmul)
{
    // The from_blob tensors do not own their storage, keep the buffers alive.
    std::vector<float> av = {1, 2, 3, 4, 5, 6};
    std::vector<float> bv = {7, 8, 9, 10, 11, 12};

    Tensor a = tinytorch::from_blob(av.data(), {2, 3});
    Tensor b = tinytorch::from_blob(bv.data(), {3, 2});
    a.set_requires_grad(true, true);
    b.set_requires_grad(true, true);

    tinytorch::backward(tinytorch::sum(tinytorch::matmul(a, b)));

    std::vector<float> gav = {15, 19, 23, 15, 19, 23};
    std::vector<float> gbv = {5, 5, 7, 7, 9, 9};

    Tensor expected_a = tinytorch::from_blob(gav.data(), {2, 3});
    Tensor expected_b = tinytorch::from_blob(gbv.data(), {3, 2});

    EXPECT_TRUE(a.grad().allclose(expected_a));
    EXPECT_TRUE(b.grad().allclose(expected_b));
}

}  // namespace

// ------------------------------------------------------------------ reshape

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchReshapeOps);
TEST_P(TinyTorchReshapeOps, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor r = tinytorch::reshape(a, {3, 2});
    TT_EXPECT_CLOSE(r, tttest::make_tensor({1, 2, 3, 4, 5, 6}, {3, 2}, device()), 0, 0);

    // reshape is a view: the gradient is the identity
    tttest::check_grads({a},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::reshape(p[0], {3, 2}));
                        },
                        1e-4, 1e-6);
}

// ------------------------------------------------------------------ repeat

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchRepeatOps);
TEST_P(TinyTorchRepeatOps, Values)
{
    // repeat requires one count per dimension
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    // tile: result[i][j] = a[i % 2][j % 2]
    Tensor r = tinytorch::repeat(a, {1, 2});
    TT_EXPECT_CLOSE(r, tttest::make_tensor({1, 2, 1, 2, 3, 4, 3, 4}, {2, 4}, device()), 0, 0);

    Tensor b = tttest::make_tensor({1, 2}, {2}, device());
    Tensor ri = tinytorch::repeat_interleave(b, 2);
    TT_EXPECT_CLOSE(ri, tttest::make_tensor({1, 1, 2, 2}, {4}, device()), 0, 0);
}

TEST_P(TinyTorchRepeatOps, RepeatInterleaveMultiDim)
{
    // repeats whole rows along dim 0: result[i][j] = a[i / 2][j]
    Tensor a  = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {3, 2}, device());
    Tensor ri = tinytorch::repeat_interleave(a, 2);
    TT_EXPECT_CLOSE(ri, tttest::make_tensor({1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6}, {6, 2}, device()), 0, 0);

    // non-contiguous input: the transposed view [[1,3,5],[2,4,6]]
    Tensor t   = a.permute_view({1, 0});
    Tensor rit = tinytorch::repeat_interleave(t, 3);
    TT_EXPECT_CLOSE(rit, tttest::make_tensor({1, 3, 5, 1, 3, 5, 1, 3, 5, 2, 4, 6, 2, 4, 6, 2, 4, 6}, {6, 3}, device()),
                    0, 0);
}

// ------------------------------------------------------------------ transpose / permute / flip

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchTransposeOps);
TEST_P(TinyTorchTransposeOps, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor t = tinytorch::transpose(a, 0, 1);
    TT_EXPECT_CLOSE(t, tttest::make_tensor({1, 4, 2, 5, 3, 6}, {3, 2}, device()), 0, 0);

    // d(sum(a^T))/da = 1
    tttest::check_grads({a},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::transpose(p[0], 0, 1));
                        },
                        1e-4, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPermuteOps);
TEST_P(TinyTorchPermuteOps, ValuesAndGradient)
{
    // 2x3x4 -> 4x3x2
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i)
    {
        data[i] = i + 1;
    }
    Tensor a = tttest::leaf(data, {2, 3, 4}, device());

    Tensor p = tinytorch::permute(a, {2, 1, 0});
    // p has sizes {4,3,2} (strides {6,2,1}); p[x, y, z] = a[z, y, x] with a strides {12,4,1}
    auto pv = tttest::to_double_vec(p);
    auto av = tttest::to_double_vec(a);
    for (int x = 0; x < 4; ++x)
    {
        for (int y = 0; y < 3; ++y)
        {
            for (int z = 0; z < 2; ++z)
            {
                EXPECT_DOUBLE_EQ(pv[x * 6 + y * 2 + z], av[z * 12 + y * 4 + x]);
            }
        }
    }

    // default (looser) tolerances: float32 finite differences of a 24-term sum
    // pick up ~1e-3 rounding noise
    tttest::check_grads({a},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::permute(p[0], {2, 1, 0}));
                        });
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchFlipOps);
TEST_P(TinyTorchFlipOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor f0 = tinytorch::flip(a, {0});
    TT_EXPECT_CLOSE(f0, tttest::make_tensor({4, 5, 6, 1, 2, 3}, {2, 3}, device()), 0, 0);

    Tensor f1 = tinytorch::flip(a, {1});
    TT_EXPECT_CLOSE(f1, tttest::make_tensor({3, 2, 1, 6, 5, 4}, {2, 3}, device()), 0, 0);
}

// ------------------------------------------------------------------ copy / fill / uniform

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchCopyOps);
TEST_P(TinyTorchCopyOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    Tensor b = tinytorch::empty({2, 2}, TensorOptions().device(device()));

    tinytorch::copy(a, b, false);
    TT_EXPECT_CLOSE(b, a, 0, 0);

    // in-place copy with dtype conversion
    Tensor c = tinytorch::empty({2, 2}, TensorOptions().device(device()).dtype(kFloat64));
    tinytorch::copy(a, c, false);
    TT_EXPECT_CLOSE(c, a, 1e-6, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchFillOps);
TEST_P(TinyTorchFillOps, Values)
{
    Tensor a = tinytorch::zeros({2, 3}, TensorOptions().device(device()));
    tinytorch::fill(a, 7.0);
    TT_EXPECT_CLOSE(a, tinytorch::full({2, 3}, 7.0f, TensorOptions().device(device())), 0, 0);

    Tensor b = tinytorch::zeros({2, 2}, TensorOptions().device(device()));
    tinytorch::fill(b, tinytorch::full({1}, 3.0f, TensorOptions().device(device())));
    TT_EXPECT_CLOSE(b, tinytorch::full({2, 2}, 3.0f, TensorOptions().device(device())), 0, 0);

    // fill along dim 1: values is indexed by the dims before dim 1,
    // i.e. c[r][col] = values[r]
    Tensor c = tinytorch::zeros({2, 3}, TensorOptions().device(device()));
    tinytorch::fill(c, tttest::make_tensor({5, 6, 7}, {3}, device()), 1);
    TT_EXPECT_CLOSE(c, tttest::make_tensor({5, 5, 5, 6, 6, 6}, {2, 3}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchUniformFillOps);
TEST_P(TinyTorchUniformFillOps, Ranges)
{
    Tensor a = tinytorch::empty({256}, TensorOptions().device(device()));
    tinytorch::uniform(a, -2.0, 2.0);
    auto v = tttest::to_double_vec(a);
    for (double x : v)
    {
        EXPECT_GE(x, -2.0);
        EXPECT_LT(x, 2.0);
    }

    Tensor b = tinytorch::empty({64}, TensorOptions().device(device()).dtype(kLong));
    tinytorch::uniform_int(b, 3, 6);
    for (double x : tttest::to_double_vec(b))
    {
        EXPECT_GE(x, 3);
        EXPECT_LE(x, 6);
        EXPECT_DOUBLE_EQ(std::floor(x), x);
    }

    // 6 sigma bounds of the sample mean (1 / sqrt(n)) and sample std (1 / sqrt(2n)) of N(0, 1)
    const int64_t n = 1 << 16;
    Tensor c        = tinytorch::empty({n}, TensorOptions().device(device()));
    tinytorch::normal_random(c);
    EXPECT_NEAR(c.mean().toDouble(), 0.0, 6.0 / std::sqrt(double(n)));
    EXPECT_NEAR(c.std().toDouble(), 1.0, 6.0 / std::sqrt(2.0 * n));
}

// ------------------------------------------------------------------ sort

TEST(TinyTorchSortOps, ValuesAndIndices)
{
    Tensor a = tttest::make_tensor({3, 1, 4, 1, 5, 9}, {2, 3}, kCPU);

    auto [s, idx] = tinytorch::sort(a, 1);
    TT_EXPECT_CLOSE(s, tttest::make_tensor({1, 3, 4, 1, 5, 9}, {2, 3}, kCPU), 0, 0);
    // indices are per-row (0-based within the reduced dim)
    TT_EXPECT_CLOSE(idx, tttest::make_tensor(std::vector<int64_t>{1, 0, 2, 0, 1, 2}, {2, 3}, kCPU), 0, 0);
}

// ------------------------------------------------------------------ clone

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchCloneOps);
TEST_P(TinyTorchCloneOps, ValuesAndIndependence)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    Tensor c = tinytorch::clone(a);
    TT_EXPECT_CLOSE(c, a, 0, 0);

    // in-place modification of the clone must not touch the original
    c.fill_(42.0);
    TT_EXPECT_CLOSE(a, tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ to / dtype

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchToOps);
TEST_P(TinyTorchToOps, DtypeConversion)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    Tensor d = a.to(kFloat64);
    EXPECT_EQ(d.dtype(), kFloat64);
    TT_EXPECT_CLOSE(d, a, 1e-6, 1e-6);

    Tensor f = d.to(kFloat);
    EXPECT_EQ(f.dtype(), kFloat);
    TT_EXPECT_CLOSE(f, a, 1e-6, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchTensorToMethod);
TEST_P(TinyTorchTensorToMethod, InPlaceDtype)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device());
    a.to_(kFloat64);
    EXPECT_EQ(a.dtype(), kFloat64);
    TT_EXPECT_CLOSE(a, tttest::make_tensor({1, 2, 3, 4}, {4}, device(), kFloat64), 1e-6, 1e-6);
}

// ------------------------------------------------------------------ slice

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSliceOps);
TEST_P(TinyTorchSliceOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {6}, device());

    TT_EXPECT_CLOSE(tinytorch::slice(a, 0, 1, 4, 1), tttest::make_tensor({2, 3, 4}, {3}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::slice(a, 0, 0, 6, 2), tttest::make_tensor({1, 3, 5}, {3}, device()), 0, 0);

    Tensor m = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    // slicing keeps the dim (size 1)
    TT_EXPECT_CLOSE(tinytorch::slice(m, 0, 1, 2, 1), tttest::make_tensor({4, 5, 6}, {1, 3}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::slice(m, 1, 0, 2, 1), tttest::make_tensor({1, 2, 4, 5}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSliceView);
TEST_P(TinyTorchSliceView, WritesThroughToOriginal)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device());
    a.slice_view(0, 1, 3).copy_(tttest::make_tensor({9, 9}, {2}, device()));
    TT_EXPECT_CLOSE(a, tttest::make_tensor({1, 9, 9, 4}, {4}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchSliceGradient);
TEST_P(TinyTorchSliceGradient, ZerosOutsideTheSlice)
{
    Tensor a = tttest::leaf({1, 2, 3, 4}, {4}, device());

    Tensor s = tinytorch::slice(a, 0, 1, 3, 1);
    tinytorch::backward(tinytorch::sum(s));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({0, 1, 1, 0}, {4}, device()), 0, 0);
}

// ------------------------------------------------------------------ stack / cat

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchStackOps);
TEST_P(TinyTorchStackOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2}, {2}, device());
    Tensor b = tttest::make_tensor({3, 4}, {2}, device());

    // stacking along a new dim 0
    Tensor s = tinytorch::stack({a, b});
    EXPECT_EQ(s.sizes(), (SizeType{2, 2}));
    TT_EXPECT_CLOSE(s, tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchCatOps);
TEST_P(TinyTorchCatOps, ValuesAndGradient)
{
    Tensor a = tttest::leaf({1, 2}, {2}, device());
    Tensor b = tttest::leaf({3, 4, 5}, {3}, device());

    Tensor c0 = tinytorch::cat({a, b}, 0);
    TT_EXPECT_CLOSE(c0, tttest::make_tensor({1, 2, 3, 4, 5}, {5}, device()), 0, 0);

    // stack along a new dim via cat
    Tensor a2 = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    Tensor b2 = tttest::make_tensor({5, 6, 7, 8}, {2, 2}, device());
    Tensor c1 = tinytorch::cat({a2, b2}, 1);
    TT_EXPECT_CLOSE(c1, tttest::make_tensor({1, 2, 5, 6, 3, 4, 7, 8}, {2, 4}, device()), 0, 0);

    tttest::check_grads({a, b},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::cat({p[0], p[1]}, 0));
                        },
                        1e-4, 1e-6);
}

// ------------------------------------------------------------------ indexing

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchIndexSelectOps);
TEST_P(TinyTorchIndexSelectOps, ValuesAndGradient)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    Tensor idx = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());

    Tensor s = tinytorch::index_select(a, 0, idx);
    TT_EXPECT_CLOSE(s, tttest::make_tensor({4, 5, 6, 1, 2, 3}, {2, 3}, device()), 0, 0);

    // out[i][j] = a[i][idx[j]] with idx = [1,0]: [[a01,a00],[a11,a10]]
    Tensor s1 = tinytorch::index_select(a, 1, idx);
    TT_EXPECT_CLOSE(s1, tttest::make_tensor({2, 1, 5, 4}, {2, 2}, device()), 0, 0);

    // dL/da: one-hot-ish scatter of the upstream gradient
    Tensor al = tttest::leaf({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    Tensor idxl = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());
    tinytorch::backward(tinytorch::sum(tinytorch::index_select(al, 0, idxl)));
    TT_EXPECT_CLOSE(al.grad(), tttest::make_tensor({1, 1, 1, 1, 1, 1}, {2, 3}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchIndexAddOps);
TEST_P(TinyTorchIndexAddOps, ValuesAndGradient)
{
    // out[row] = a[row] + a[index[row]]  (index maps rows onto each other)
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    Tensor idx = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());

    Tensor r = tinytorch::index_add(a, 0, idx, a);
    TT_EXPECT_CLOSE(r, tttest::make_tensor({4, 6, 4, 6}, {2, 2}, device()), 0, 0);

    Tensor al = tttest::leaf({1, 2, 3, 4}, {2, 2}, device());
    Tensor idxl = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());
    tinytorch::backward(tinytorch::sum(tinytorch::index_add(al, 0, idxl, al)));
    // each row contributes once as source and once as target -> 2 everywhere
    TT_EXPECT_CLOSE(al.grad(), tttest::make_tensor({2, 2, 2, 2}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchIndexCopyOps);
TEST_P(TinyTorchIndexCopyOps, Values)
{
    Tensor target = tinytorch::zeros({2, 2}, TensorOptions().device(device()));
    Tensor source = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    Tensor idx = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());

    tinytorch::index_copy(target, 0, idx, source);
    TT_EXPECT_CLOSE(target, tttest::make_tensor({3, 4, 1, 2}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchGatherOps);
TEST_P(TinyTorchGatherOps, Values)
{
    // out[i][j] = data[i][index[i][j]]  (dim == 1)
    Tensor data = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    Tensor idx = tttest::make_tensor(std::vector<int64_t>{2, 0, 1, 2}, {2, 2}, device());

    // out[i][j] = data[i][idx[i][j]]: [[1,2,3],[4,5,6]] with [[2,0],[1,2]] -> [[3,1],[5,6]]
    Tensor g = tinytorch::gather(data, 1, idx);
    TT_EXPECT_CLOSE(g, tttest::make_tensor({3, 1, 5, 6}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ padding

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPadding2dZero);
TEST_P(TinyTorchPadding2dZero, Values)
{
    // padding_2d requires 4D input with batch/channel == 1
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());

    Tensor p = tinytorch::padding_2d(a, 1, 1, 1, 1, kZero);
    EXPECT_EQ(p.sizes(), (SizeType{1, 1, 4, 4}));
    TT_EXPECT_CLOSE(p, tttest::make_tensor({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0}, {1, 1, 4, 4}, device()),
                    0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPadding2dBorder);
TEST_P(TinyTorchPadding2dBorder, Values)
{
    // kBorder replicates the edge values (clamp)
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());

    Tensor p = tinytorch::padding_2d(a, 1, 1, 1, 1, kBorder);
    TT_EXPECT_CLOSE(p,
                    tttest::make_tensor({1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}, {1, 1, 4, 4}, device()), 0,
                    0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPadding2dReflect);
TEST_P(TinyTorchPadding2dReflect, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());

    Tensor p = tinytorch::padding_2d(a, 1, 1, 1, 1, kReflect);
    // reflection maps -1 -> 1 and 2 -> 0, so each axis gives indices 1,0,1,0:
    // row0 = a[1] reflected = [4,3,4,3], row1 = a[0] reflected = [2,1,2,1], etc.
    TT_EXPECT_CLOSE(p,
                    tttest::make_tensor({4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1}, {1, 1, 4, 4}, device()), 0,
                    0);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPadding3dZero);
TEST_P(TinyTorchPadding3dZero, Values)
{
    // padding_3d requires 5D input with batch/channel == 1
    std::vector<float> a(2 * 2 * 2);
    for (int i = 0; i < 8; ++i)
    {
        a[i] = i + 1;
    }
    Tensor t = tttest::make_tensor(a, {1, 1, 2, 2, 2}, device());

    Tensor p = tinytorch::padding_3d(t, 1, 1, 1, 1, 1, 1, kZero);
    EXPECT_EQ(p.sizes(), (SizeType{1, 1, 4, 4, 4}));

    // interior (slice 1..3 on the three spatial dims) must equal the original
    Tensor interior = p.slice(2, 1, 3).slice(3, 1, 3).slice(4, 1, 3);
    TT_EXPECT_CLOSE(interior, t, 0, 0);

    // border (front spatial slab) must be zero
    Tensor border = p.slice(2, 0, 1);
    TT_EXPECT_CLOSE(border, tinytorch::zeros({1, 1, 1, 4, 4}, TensorOptions().device(device())), 0, 0);
}

namespace
{
std::vector<float> test_values(int64_t n, double seed)
{
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i)
    {
        v[i] = (float)std::sin(seed + 0.37 * (double)i);
    }
    return v;
}

// pads every [b, c] slice on its own; the single-slice path is the reference for the batched/strided one
Tensor padding_2d_per_slice(Tensor x, int l, int r, int t, int b, tinytorch::PaddingMode mode)
{
    std::vector<Tensor> rows;
    for (int64_t n = 0; n < x.size(0); ++n)
    {
        std::vector<Tensor> channels;
        for (int64_t c = 0; c < x.size(1); ++c)
        {
            Tensor slice = x.slice(0, n, n + 1).slice(1, c, c + 1).clone();
            channels.push_back(tinytorch::padding_2d(slice, l, r, t, b, mode));
        }
        rows.push_back(tinytorch::cat(channels, 1));
    }
    return tinytorch::cat(rows, 0);
}

Tensor padding_3d_per_slice(Tensor x, int l, int r, int t, int b, int f, int k, tinytorch::PaddingMode mode)
{
    std::vector<Tensor> rows;
    for (int64_t n = 0; n < x.size(0); ++n)
    {
        std::vector<Tensor> channels;
        for (int64_t c = 0; c < x.size(1); ++c)
        {
            Tensor slice = x.slice(0, n, n + 1).slice(1, c, c + 1).clone();
            channels.push_back(tinytorch::padding_3d(slice, l, r, t, b, f, k, mode));
        }
        rows.push_back(tinytorch::cat(channels, 1));
    }
    return tinytorch::cat(rows, 0);
}
}  // namespace

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPaddingBatched);
TEST_P(TinyTorchPaddingBatched, Padding2dBatchAndStrides)
{
    for (auto mode : {kZero, kBorder, kReflect})
    {
        // batch 2, 3 channels
        Tensor x = tttest::make_tensor(test_values(2 * 3 * 4 * 5, 0.1), {2, 3, 4, 5}, device());
        TT_EXPECT_CLOSE(tinytorch::padding_2d(x, 1, 2, 2, 1, mode), padding_2d_per_slice(x, 1, 2, 2, 1, mode), 0, 0);

        // non-contiguous: channels/batch swapped and every second column
        Tensor base = tttest::make_tensor(test_values(3 * 2 * 4 * 10, 0.2), {3, 2, 4, 10}, device());
        Tensor nc   = base.permute_view({1, 0, 2, 3}).slice_view(3, 0, 10, 2);
        ASSERT_FALSE(nc.is_contiguous());
        TT_EXPECT_CLOSE(tinytorch::padding_2d(nc, 2, 1, 1, 2, mode), padding_2d_per_slice(nc, 2, 1, 1, 2, mode), 0, 0);

        // the output overload with a non-contiguous result
        Tensor xs       = x.slice_view(1, 1, 3, 1);
        Tensor out_base = tinytorch::zeros({2, 2, 7, 14}, TensorOptions().device(device()));
        Tensor out      = out_base.slice_view(3, 0, 14, 2);
        tinytorch::padding_2d(xs, out, 1, 1, 2, 1, mode);
        TT_EXPECT_CLOSE(out, padding_2d_per_slice(xs, 1, 1, 2, 1, mode), 0, 0);
    }
}

TEST_P(TinyTorchPaddingBatched, Padding3dBatchAndStrides)
{
    for (auto mode : {kZero, kBorder, kReflect})
    {
        Tensor x = tttest::make_tensor(test_values(2 * 2 * 3 * 4 * 3, 0.3), {2, 2, 3, 4, 3}, device());
        TT_EXPECT_CLOSE(tinytorch::padding_3d(x, 1, 2, 1, 0, 2, 1, mode),
                        padding_3d_per_slice(x, 1, 2, 1, 0, 2, 1, mode), 0, 0);

        Tensor base = tttest::make_tensor(test_values(2 * 2 * 3 * 4 * 6, 0.4), {2, 2, 3, 4, 6}, device());
        Tensor nc   = base.permute_view({1, 0, 2, 3, 4}).slice_view(4, 0, 6, 2);
        ASSERT_FALSE(nc.is_contiguous());
        TT_EXPECT_CLOSE(tinytorch::padding_3d(nc, 1, 1, 1, 1, 1, 1, mode),
                        padding_3d_per_slice(nc, 1, 1, 1, 1, 1, 1, mode), 0, 0);
    }
}

// ------------------------------------------------------------------ matmul

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMatmulOps);
TEST_P(TinyTorchMatmulOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    Tensor b = tttest::make_tensor({7, 8, 9, 10, 11, 12}, {3, 2}, device());

    Tensor m = tinytorch::matmul(a, b);
    TT_EXPECT_CLOSE(m, tttest::make_tensor({58, 64, 139, 154}, {2, 2}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMatmulBatched);
TEST_P(TinyTorchMatmulBatched, ValuesAndGradient)
{
    // two 2x2 @ 2x2 matrices stacked along dim 0
    Tensor a = tttest::leaf({1, 2, 3, 4, 10, 20, 30, 40}, {2, 2, 2}, device());
    Tensor b = tttest::leaf({1, 0, 0, 1, 5, 0, 0, 1}, {2, 2, 2}, device());

    Tensor m = tinytorch::matmul(a, b);
    TT_EXPECT_CLOSE(m, tttest::make_tensor({1, 2, 3, 4, 50, 20, 150, 40}, {2, 2, 2}, device()), 1e-5, 1e-6);

    tttest::check_grads({a, b},
                        [](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::matmul(p[0], p[1]));
                        },
                        2e-2, 2e-3);
}

namespace
{
// Reference batched matmul with broadcast batch dims over row-major data (both operands >= 2-D).
std::vector<double> ref_matmul(const std::vector<double>& a, std::vector<int64_t> as, const std::vector<double>& b,
                               std::vector<int64_t> bs, std::vector<int64_t>& out_sizes)
{
    int64_t batch = (int64_t)std::max(as.size(), bs.size()) - 2;
    while ((int64_t)as.size() < batch + 2) as.insert(as.begin(), 1);
    while ((int64_t)bs.size() < batch + 2) bs.insert(bs.begin(), 1);
    out_sizes.assign(batch + 2, 1);
    for (int64_t d = 0; d < batch; ++d)
    {
        out_sizes[d] = std::max(as[d], bs[d]);
    }
    int64_t M = as[batch], K = as[batch + 1], N = bs[batch + 1];
    out_sizes[batch]     = M;
    out_sizes[batch + 1] = N;

    int64_t num_batches = 1;
    for (int64_t d = 0; d < batch; ++d) num_batches *= out_sizes[d];

    std::vector<double> out(num_batches * M * N, 0);
    for (int64_t bi = 0; bi < num_batches; ++bi)
    {
        int64_t rem = bi, a_off = 0, b_off = 0, a_stride = M * K, b_stride = K * N;
        for (int64_t d = batch - 1; d >= 0; --d)
        {
            int64_t idx = rem % out_sizes[d];
            rem /= out_sizes[d];
            a_off += (as[d] == 1 ? 0 : idx) * a_stride;
            b_off += (bs[d] == 1 ? 0 : idx) * b_stride;
            a_stride *= as[d];
            b_stride *= bs[d];
        }
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n)
                for (int64_t k = 0; k < K; ++k)
                    out[(bi * M + m) * N + n] += a[a_off + m * K + k] * b[b_off + k * N + n];
    }
    return out;
}

struct MatmulCase
{
    std::vector<int64_t> a, b;
};

const std::vector<MatmulCase> kMatmulCases = {
    {{2, 3, 4}, {4, 5}},           // 3-D @ 2-D
    {{3, 4}, {2, 4, 5}},           // 2-D @ 3-D
    {{2, 3, 4}, {2, 4, 5}},        // equal batch
    {{1, 3, 4}, {2, 4, 2}},        // broadcast size-1 batch of a
    {{2, 1, 3, 4}, {1, 3, 4, 2}},  // broadcast in both
    {{2, 2, 3, 4}, {4, 3}},        // 4-D @ 2-D
};

int64_t numel_of(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto v : s) n *= v;
    return n;
}
}  // namespace

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchMatmulBroadcast);
TEST_P(TinyTorchMatmulBroadcast, Values)
{
    for (const auto& c : kMatmulCases)
    {
        auto av = test_values(numel_of(c.a), 0.5);
        auto bv = test_values(numel_of(c.b), 1.5);
        std::vector<int64_t> out_sizes;
        auto ref = ref_matmul({av.begin(), av.end()}, c.a, {bv.begin(), bv.end()}, c.b, out_sizes);

        Tensor m = tinytorch::matmul(tttest::make_tensor(av, c.a, device()), tttest::make_tensor(bv, c.b, device()));
        std::vector<float> reff(ref.begin(), ref.end());
        TT_EXPECT_CLOSE(m, tttest::make_tensor(reff, out_sizes, device()), 1e-5, 1e-5);
    }
}

TEST_P(TinyTorchMatmulBroadcast, Gradient)
{
    for (const auto& c : kMatmulCases)
    {
        Tensor a = tttest::leaf(test_values(numel_of(c.a), 0.5), c.a, device());
        Tensor b = tttest::leaf(test_values(numel_of(c.b), 1.5), c.b, device());
        std::vector<int64_t> out_sizes;
        ref_matmul(std::vector<double>(numel_of(c.a)), c.a, std::vector<double>(numel_of(c.b)), c.b, out_sizes);
        Tensor w = tttest::make_tensor(test_values(numel_of(out_sizes), 2.5), out_sizes, device());

        tttest::check_grads(
            {a, b}, [w](const std::vector<Tensor>& p) { return tinytorch::sum(tinytorch::matmul(p[0], p[1]) * w); },
            2e-2, 2e-3);
    }
}

TEST_P(TinyTorchMatmulBroadcast, NonContiguous)
{
    // a: every second column of a [2, 3, 8] tensor -> [2, 3, 4]; b: transposed [5, 4] -> [4, 5]
    auto av  = test_values(2 * 3 * 8, 0.5);
    auto bv  = test_values(5 * 4, 1.5);
    Tensor a = tttest::make_tensor(av, {2, 3, 8}, device()).slice_view(2, 0, 8, 2);
    Tensor b = tttest::make_tensor(bv, {5, 4}, device()).permute_view({1, 0});
    ASSERT_FALSE(a.is_contiguous());
    ASSERT_FALSE(b.is_contiguous());

    std::vector<double> ac, bc;
    for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 4; ++k) ac.push_back(av[i * 8 + 2 * k]);
    for (int k = 0; k < 4; ++k)
        for (int n = 0; n < 5; ++n) bc.push_back(bv[n * 4 + k]);
    std::vector<int64_t> out_sizes;
    auto ref = ref_matmul(ac, {2, 3, 4}, bc, {4, 5}, out_sizes);
    std::vector<float> reff(ref.begin(), ref.end());
    TT_EXPECT_CLOSE(tinytorch::matmul(a, b), tttest::make_tensor(reff, out_sizes, device()), 1e-5, 1e-5);

    // gradients through strided views
    Tensor al = tttest::leaf(av, {2, 3, 8}, device());
    Tensor bl = tttest::leaf(test_values(4 * 10, 1.5), {4, 10}, device());
    Tensor w  = tttest::make_tensor(test_values(2 * 3 * 5, 2.5), {2, 3, 5}, device());
    tttest::check_grads(
        {al, bl}, [w](const std::vector<Tensor>& p)
        { return tinytorch::sum(tinytorch::matmul(p[0].slice(2, 0, 8, 2), p[1].slice(1, 0, 10, 2)) * w); }, 2e-2, 2e-3);
}

TEST_P(TinyTorchMatmulBroadcast, OneDimensional)
{
    auto av = test_values(4, 0.5);
    auto bv = test_values(2 * 4 * 3, 1.5);
    std::vector<int64_t> out_sizes;

    // [4] @ [2, 4, 3] -> [2, 3]
    auto ref = ref_matmul({av.begin(), av.end()}, {1, 4}, {bv.begin(), bv.end()}, {2, 4, 3}, out_sizes);
    Tensor m = tinytorch::matmul(tttest::make_tensor(av, {4}, device()), tttest::make_tensor(bv, {2, 4, 3}, device()));
    TT_EXPECT_CLOSE(m, tttest::make_tensor(std::vector<float>(ref.begin(), ref.end()), {2, 3}, device()), 1e-5, 1e-5);

    // [2, 3, 4] @ [4] -> [2, 3]
    ref = ref_matmul({bv.begin(), bv.end()}, {2, 3, 4}, {av.begin(), av.end()}, {4, 1}, out_sizes);
    m   = tinytorch::matmul(tttest::make_tensor(bv, {2, 3, 4}, device()), tttest::make_tensor(av, {4}, device()));
    TT_EXPECT_CLOSE(m, tttest::make_tensor(std::vector<float>(ref.begin(), ref.end()), {2, 3}, device()), 1e-5, 1e-5);

    // [4] @ [4] -> dot product
    double dot = 0;
    for (int i = 0; i < 4; ++i) dot += (double)av[i] * bv[i];
    m = tinytorch::matmul(tttest::make_tensor(av, {4}, device()),
                          tttest::make_tensor(std::vector<float>(bv.begin(), bv.begin() + 4), {4}, device()));
    EXPECT_EQ(m.numel(), 1);
    EXPECT_NEAR(m.toFloat(), dot, 1e-5);

    Tensor a = tttest::leaf(av, {4}, device());
    Tensor b = tttest::leaf(bv, {2, 4, 3}, device());
    Tensor w = tttest::make_tensor(test_values(6, 2.5), {2, 3}, device());
    tttest::check_grads(
        {a, b}, [w](const std::vector<Tensor>& p) { return tinytorch::sum(tinytorch::matmul(p[0], p[1]) * w); }, 2e-2,
        2e-3);
}

TEST_P(TinyTorchMatmulBroadcast, HalfAccumulatesInFloat)
{
    // 4096 * 1 is exact in half, but a half accumulator gets stuck at 2048 (2048 + 1 rounds to 2048)
    std::vector<float> ones(4096, 1.0f);
    Tensor a = tttest::make_tensor(ones, {1, 4096}, device(), tinytorch::kHalf);
    Tensor b = tttest::make_tensor(ones, {4096, 1}, device(), tinytorch::kHalf);
    EXPECT_EQ(tinytorch::matmul(a, b).to(kFloat).toFloat(), 4096.0f);
}

// ------------------------------------------------------------------ conv2d

TEST(TinyTorchConv2d, BoxFilter)
{
    // 4x4 input with distinct values, 3x3 all-ones kernel, padding 1 -> 4x4 output
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> w(9, 1.0f);

    Tensor input = tinytorch::from_blob(data.data(), {1, 1, 4, 4});
    Tensor weight = tinytorch::from_blob(w.data(), {1, 1, 3, 3});

    Tensor out = tinytorch::conv2d(input, weight, Tensor(), 1, 1, 1, 1);
    EXPECT_EQ(out.sizes(), (SizeType{1, 1, 4, 4}));

    // out[0][0] = zero-padded 3x3 window at (0,0): 1+2+5+6 = 14
    // out[1][1] = full 3x3 window at (1,1): 1+2+3+5+6+7+9+10+11 = 54
    auto v = tttest::to_double_vec(out);
    EXPECT_DOUBLE_EQ(v[0], 14);
    EXPECT_DOUBLE_EQ(v[5], 54);
}

// conv2d must produce the same result on CPU and CUDA
TT_INSTANTIATE_DEVICE_TESTS(TinyTorchConv2dDevice);
TEST_P(TinyTorchConv2dDevice, BoxFilter)
{
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> w(9, 1.0f);

    Tensor input = tttest::make_tensor(data, {1, 1, 4, 4}, device());
    Tensor weight = tttest::make_tensor(w, {1, 1, 3, 3}, device());

    Tensor out = tinytorch::conv2d(input, weight, Tensor(), 1, 1, 1, 1);
    EXPECT_EQ(out.sizes(), (SizeType{1, 1, 4, 4}));

    // out[0][0] = zero-padded 3x3 window at (0,0): 1+2+5+6 = 14
    // out[1][1] = full 3x3 window at (1,1): 1+2+3+5+6+7+9+10+11 = 54
    auto v = tttest::to_double_vec(out);
    EXPECT_NEAR(v[0], 14, 1e-4);
    EXPECT_NEAR(v[5], 54, 1e-4);
}

TEST_P(TinyTorchConv2dDevice, ValidConvolution)
{
    // without padding the output is the 'valid' cross-correlation: out[y][x] = sum_ij in[y+i][x+j] * w[i][j]
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> w    = {0, 0, 0, 0, 0, 0, 0, 0, 1};

    Tensor out = tinytorch::conv2d(tttest::make_tensor(data, {1, 1, 4, 4}, device()),
                                   tttest::make_tensor(w, {1, 1, 3, 3}, device()));
    TT_EXPECT_CLOSE(out, tttest::make_tensor({11, 12, 15, 16}, {1, 1, 2, 2}, device()), 0, 0);
}

namespace
{
// Reference conv2d (PyTorch semantics, zero padding) over row-major data.
std::vector<float> ref_conv2d(const std::vector<float>& x, const std::vector<int64_t>& xs, const std::vector<float>& w,
                              const std::vector<int64_t>& ws, const std::vector<float>& bias, int stride, int pad,
                              int dil, int groups, std::vector<int64_t>& out_sizes)
{
    int64_t N = xs[0], C = xs[1], H = xs[2], W = xs[3];
    int64_t CO = ws[0], CI = ws[1], KH = ws[2], KW = ws[3];
    int64_t OH = (H + 2 * pad - dil * (KH - 1) - 1) / stride + 1;
    int64_t OW = (W + 2 * pad - dil * (KW - 1) - 1) / stride + 1;
    out_sizes  = {N, CO, OH, OW};
    std::vector<float> out(N * CO * OH * OW);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t co = 0; co < CO; ++co)
            for (int64_t oy = 0; oy < OH; ++oy)
                for (int64_t ox = 0; ox < OW; ++ox)
                {
                    double sum = bias.empty() ? 0.0 : bias[co];
                    for (int64_t ci = 0; ci < CI; ++ci)
                        for (int64_t ky = 0; ky < KH; ++ky)
                            for (int64_t kx = 0; kx < KW; ++kx)
                            {
                                int64_t iy = oy * stride - pad + ky * dil;
                                int64_t ix = ox * stride - pad + kx * dil;
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                                int64_t c = co / (CO / groups) * CI + ci;
                                sum += (double)x[((n * C + c) * H + iy) * W + ix] *
                                       w[((co * CI + ci) * KH + ky) * KW + kx];
                            }
                    out[((n * CO + co) * OH + oy) * OW + ox] = (float)sum;
                }
    return out;
}

struct ConvCase
{
    std::vector<int64_t> x, w;
    bool bias;
    int stride, pad, dil, groups;
};

const std::vector<ConvCase> kConvCases = {
    {{2, 3, 5, 6}, {4, 3, 3, 2}, true, 1, 1, 1, 1},   // batch, Cin, Cout, padding, non-square kernel
    {{1, 2, 4, 4}, {1, 2, 3, 3}, false, 1, 0, 1, 1},  // valid convolution
    {{2, 4, 7, 7}, {2, 2, 3, 3}, true, 2, 1, 2, 2},   // stride, dilation, groups
};
}  // namespace

TEST_P(TinyTorchConv2dDevice, ValuesBatchedMultiChannel)
{
    for (const auto& c : kConvCases)
    {
        auto xv = test_values(numel_of(c.x), 0.5);
        auto wv = test_values(numel_of(c.w), 1.5);
        auto bv = c.bias ? test_values(c.w[0], 2.5) : std::vector<float>();
        std::vector<int64_t> out_sizes;
        auto ref = ref_conv2d(xv, c.x, wv, c.w, bv, c.stride, c.pad, c.dil, c.groups, out_sizes);

        Tensor bias = c.bias ? tttest::make_tensor(bv, {c.w[0]}, device()) : Tensor();
        Tensor out  = tinytorch::conv2d(tttest::make_tensor(xv, c.x, device()), tttest::make_tensor(wv, c.w, device()),
                                        bias, c.stride, c.pad, c.dil, c.groups);
        TT_EXPECT_CLOSE(out, tttest::make_tensor(ref, out_sizes, device()), 1e-5, 1e-5);
    }
}

TEST_P(TinyTorchConv2dDevice, Gradient)
{
    for (const auto& c : kConvCases)
    {
        Tensor x                   = tttest::leaf(test_values(numel_of(c.x), 0.5), c.x, device());
        Tensor w                   = tttest::leaf(test_values(numel_of(c.w), 1.5), c.w, device());
        std::vector<Tensor> params = {x, w};
        if (c.bias)
        {
            params.push_back(tttest::leaf(test_values(c.w[0], 2.5), {c.w[0]}, device()));
        }
        std::vector<int64_t> out_sizes;
        ref_conv2d(std::vector<float>(numel_of(c.x)), c.x, std::vector<float>(numel_of(c.w)), c.w, {}, c.stride, c.pad,
                   c.dil, c.groups, out_sizes);
        Tensor g = tttest::make_tensor(test_values(numel_of(out_sizes), 3.5), out_sizes, device());

        tttest::check_grads(
            params,
            [g, c](const std::vector<Tensor>& p)
            {
                Tensor bias = p.size() > 2 ? p[2] : Tensor();
                return tinytorch::sum(tinytorch::conv2d(p[0], p[1], bias, c.stride, c.pad, c.dil, c.groups) * g);
            },
            2e-2, 2e-3);
    }
}

TEST_P(TinyTorchConv2dDevice, NonContiguousInput)
{
    // input: every second column of [2, 3, 5, 12]; weight: [3, 4, 3, 3] permuted to [4, 3, 3, 3]
    auto xv  = test_values(2 * 3 * 5 * 12, 0.5);
    auto wv  = test_values(3 * 4 * 3 * 3, 1.5);
    Tensor x = tttest::make_tensor(xv, {2, 3, 5, 12}, device()).slice_view(3, 0, 12, 2);
    Tensor w = tttest::make_tensor(wv, {3, 4, 3, 3}, device()).permute_view({1, 0, 2, 3});
    ASSERT_FALSE(x.is_contiguous());
    ASSERT_FALSE(w.is_contiguous());

    std::vector<float> xc, wc;
    for (int i = 0; i < 2 * 3 * 5; ++i)
        for (int j = 0; j < 6; ++j) xc.push_back(xv[i * 12 + 2 * j]);
    for (int co = 0; co < 4; ++co)
        for (int ci = 0; ci < 3; ++ci)
            for (int k = 0; k < 9; ++k) wc.push_back(wv[(ci * 4 + co) * 9 + k]);
    std::vector<int64_t> out_sizes;
    auto ref = ref_conv2d(xc, {2, 3, 5, 6}, wc, {4, 3, 3, 3}, {}, 1, 1, 1, 1, out_sizes);
    TT_EXPECT_CLOSE(tinytorch::conv2d(x, w, Tensor(), 1, 1), tttest::make_tensor(ref, out_sizes, device()), 1e-5, 1e-5);

    Tensor xl = tttest::leaf(xv, {2, 3, 5, 12}, device());
    Tensor wl = tttest::leaf(test_values(4 * 3 * 3 * 6, 1.5), {4, 3, 3, 6}, device());
    Tensor g  = tttest::make_tensor(test_values(numel_of(out_sizes), 3.5), out_sizes, device());
    tttest::check_grads(
        {xl, wl},
        [g](const std::vector<Tensor>& p)
        {
            return tinytorch::sum(tinytorch::conv2d(p[0].slice(3, 0, 12, 2), p[1].slice(3, 0, 6, 2), Tensor(), 1, 1) *
                                  g);
        },
        2e-2, 2e-3);
}

// ------------------------------------------------------------------ TensorInfo

TEST(TinyTorchTensorInfo, DimIndexAccessOnStridedTensor)
{
    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    Tensor t                = tinytorch::from_blob(data.data(), {2, 3}).permute_view({1, 0});  // [[0,3],[1,4],[2,5]]
    ASSERT_FALSE(t.is_contiguous());
    tinytorch::TensorInfo<float> info(t);

    for (int64_t i = 0; i < 6; ++i)
    {
        auto idx       = info.LinearIndexToDimIndex(i);
        float expected = data[idx[1] * 3 + idx[0]];
        EXPECT_EQ(info(idx), expected) << i;
        EXPECT_EQ(info[idx], expected) << i;
        EXPECT_EQ(info[i], expected) << i;
        EXPECT_TRUE(info.index_in_range(idx));
    }

    auto idx = info.LinearIndexToDimIndex(0);
    idx[0]   = 3;
    EXPECT_FALSE(info.index_in_range(idx));
    idx[0] = 2;
    idx[1] = 2;
    EXPECT_FALSE(info.index_in_range(idx));

    tinytorch::DimIndexStruct<2, int64_t> full(std::vector<int64_t>{1, 2});
    EXPECT_EQ(full[0], 1);
    EXPECT_EQ(full[1], 2);
}

// ------------------------------------------------------------------ poisson noise

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchPoissonNoise);
TEST_P(TinyTorchPoissonNoise, RunsAndModifies)
{
    Tensor a = tinytorch::full({16}, 5.0f, TensorOptions().device(device()));
    tinytorch::add_poisson_noise_(a);
    EXPECT_EQ(a.sizes(), (SizeType{16}));
    // the noise is additive on average; just make sure values stay finite and positive-ish
    for (double x : tttest::to_double_vec(a))
    {
        EXPECT_TRUE(std::isfinite(x));
    }
}

// ------------------------------------------------------------------ grid_sample

namespace
{

// input (1,1,2,2) = [[1,2],[3,4]]; grid (1,H,W,2) with u=grid[...,0], v=grid[...,1] in [-1,1]
Tensor make_grid(const std::vector<float>& uv, Device d)
{
    // uv: interleaved (u0, v0, u1, v1, ...)
    std::vector<float> g(uv.size() * 2);
    for (size_t i = 0; i < uv.size() / 2; ++i)
    {
        g[2 * i]     = uv[2 * i];
        g[2 * i + 1] = uv[2 * i + 1];
    }
    return tttest::make_tensor(g, {1, 1, (int64_t)(uv.size() / 2), 2}, d);
}

}  // namespace

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchGridSample2d);
TEST_P(TinyTorchGridSample2d, BilinearAlignCorners)
{
    Tensor input = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());
    // corners: (-1,-1) -> px(0,0) -> 1,  (1,-1) -> px(0,1) -> 2,  (-1,1) -> px(1,0) -> 3,  (1,1) -> px(1,1) -> 4
    Tensor grid = make_grid({-1, -1, 1, -1, -1, 1, 1, 1}, device());

    // note: the grid layout is (N,H,W,2); make_grid builds {1,1,4,2}, so the
    // sampled output has shape {1,1,1,4}
    Tensor out = tinytorch::nn::functional::grid_sample(
        input, grid, tinytorch::nn::functional::GridSampleFuncOptions().mode(kBilinear).padding_mode(kBorder).align_corners(true));

    TT_EXPECT_CLOSE(out, tttest::make_tensor({1, 2, 3, 4}, {1, 1, 1, 4}, device()), 1e-5, 1e-6);
}

TEST_P(TinyTorchGridSample2d, BilinearCenter)
{
    Tensor input = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());
    // center: (0,0) -> px(0.5,0.5) -> (1+2+3+4)/4 = 2.5
    Tensor grid = make_grid({0, 0}, device());

    Tensor out = tinytorch::nn::functional::grid_sample(
        input, grid, tinytorch::nn::functional::GridSampleFuncOptions().mode(kBilinear).padding_mode(kBorder).align_corners(true));

    TT_EXPECT_CLOSE(out, tttest::make_tensor({2.5f}, {1, 1, 1, 1}, device()), 1e-5, 1e-6);
}

TEST_P(TinyTorchGridSample2d, Nearest)
{
    Tensor input = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());
    // (0.5,0.5) -> px(0.5,0.5); nearest rounds to (1,1) -> 4 (bilinear would give 2.5)
    Tensor grid = make_grid({0.5f, 0.5f}, device());

    Tensor out = tinytorch::nn::functional::grid_sample(
        input, grid, tinytorch::nn::functional::GridSampleFuncOptions().mode(kNearest).padding_mode(kBorder).align_corners(true));

    TT_EXPECT_CLOSE(out, tttest::make_tensor({4.0f}, {1, 1, 1, 1}, device()), 0, 0);
}

TEST_P(TinyTorchGridSample2d, Gradient)
{
    Tensor input = tttest::leaf({1, 2, 3, 4}, {1, 1, 2, 2}, device());

    // 4 sample points (u,v) interleaved -> grid {1,1,4,2}, made a leaf
    std::vector<float> uv = {-0.5, 0.5, 0.5, -0.5, 0.25, -0.25, -0.25, 0.25};
    std::vector<float> g(uv.size() * 2);
    for (size_t i = 0; i < uv.size() / 2; ++i)
    {
        g[2 * i]     = uv[2 * i];
        g[2 * i + 1] = uv[2 * i + 1];
    }
    Tensor grid = tttest::leaf(g, {1, 1, 4, 2}, device());

    auto opts =
        tinytorch::nn::functional::GridSampleFuncOptions().mode(kBilinear).padding_mode(kBorder).align_corners(true);

    // check both the input gradient and the grid gradient against finite differences
    tttest::check_grads({input, grid},
                        [&opts](const std::vector<Tensor>& p)
                        {
                            return tinytorch::sum(tinytorch::nn::functional::grid_sample(p[0], p[1], opts));
                        });
}

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchGridSample3d);
TEST_P(TinyTorchGridSample3d, BilinearAndNearest)
{
    // input (1,1,2,2,2) with values 1..8; grid layout (N,D,H,W,3)
    std::vector<float> data(8);
    for (int i = 0; i < 8; ++i)
    {
        data[i] = i + 1;
    }
    Tensor input = tttest::make_tensor(data, {1, 1, 2, 2, 2}, device());

    // 3 sample points: (-1,-1,-1) -> 1, (1,1,1) -> 8, (0,0,0) -> center 4.5
    Tensor grid = tttest::make_tensor({-1, -1, -1, 1, 1, 1, 0, 0, 0}, {1, 1, 3, 1, 3}, device());

    auto opts = tinytorch::nn::functional::GridSampleFuncOptions().mode(kBilinear).padding_mode(kBorder).align_corners(true);
    Tensor out = tinytorch::nn::functional::grid_sample(input, grid, opts);
    TT_EXPECT_CLOSE(out, tttest::make_tensor({1, 8, 4.5}, {1, 1, 1, 3, 1}, device()), 1e-5, 1e-6);

    // nearest: (0.2,0.2,0.2) -> px(0.6,0.6,0.6) -> rounds to (1,1,1) -> 8 (bilinear would give 5.2)
    Tensor grid_n = tttest::make_tensor({0.2, 0.2, 0.2}, {1, 1, 1, 1, 3}, device());
    auto opts_n   = tinytorch::nn::functional::GridSampleFuncOptions().mode(kNearest).padding_mode(kBorder).align_corners(true);
    Tensor out_n  = tinytorch::nn::functional::grid_sample(input, grid_n, opts_n);
    TT_EXPECT_CLOSE(out_n, tttest::make_tensor({8.0f}, {1, 1, 1, 1, 1}, device()), 0, 0);
}

// samples on and beyond the border must be clipped to the volume in nearest mode
TEST_P(TinyTorchGridSample3d, NearestOutsideIsClipped)
{
    std::vector<float> data(8);
    for (int i = 0; i < 8; ++i)
    {
        data[i] = i + 1;
    }
    Tensor input = tttest::leaf(data, {1, 1, 2, 2, 2}, device());
    Tensor grid  = tttest::make_tensor({1, 1, 1, 3, 3, 3, -3, -3, -3}, {1, 1, 3, 1, 3}, device());

    auto opts =
        tinytorch::nn::functional::GridSampleFuncOptions().mode(kNearest).padding_mode(kBorder).align_corners(false);
    Tensor out = tinytorch::nn::functional::grid_sample(input, grid, opts);
    TT_EXPECT_CLOSE(out, tttest::make_tensor({8, 8, 1}, {1, 1, 1, 3, 1}, device()), 0, 0);

    tinytorch::backward(tinytorch::sum(out));
    TT_EXPECT_CLOSE(input.grad(), tttest::make_tensor({1, 0, 0, 0, 0, 0, 0, 2}, {1, 1, 2, 2, 2}, device()), 0, 0);
}

namespace
{
using tinytorch::nn::functional::GridSampleFuncOptions;

GridSampleFuncOptions gs_opts(tinytorch::InterpolationType mode, tinytorch::PaddingMode padding, bool align_corners)
{
    return GridSampleFuncOptions().mode(mode).padding_mode(padding).align_corners(align_corners);
}

// inverse of the grid_sample unnormalization: pixel coordinate -> grid value in [-1,1]
double gs_pixel_to_grid(double x, int64_t size, bool align_corners)
{
    if (align_corners)
    {
        return size > 1 ? 2.0 * x / double(size - 1) - 1.0 : 0.0;
    }
    return (2.0 * x + 1.0) / double(size) - 1.0;
}

// grid (1,1,n,2) sampling a (1,1,1,4) row at the pixel x-coordinates xs (y = 0)
Tensor gs_row_grid(const std::vector<double>& xs, bool align_corners, Device d, ScalarType dtype = kFloat)
{
    std::vector<float> g;
    for (double x : xs)
    {
        g.push_back((float)gs_pixel_to_grid(x, 4, align_corners));
        g.push_back((float)gs_pixel_to_grid(0, 1, align_corners));
    }
    return tttest::make_tensor(g, {1, 1, (int64_t)xs.size(), 2}, d, dtype);
}

// pixel coordinates that are inside, near the border and outside but at least 0.1 px away from every kink
// (integers, clip and reflection boundaries, rounding ties of nearest)
const std::vector<double> gs_check_coords = {-3.3, -0.7, -0.2, 0.3, 1.6, 2.7, 3.2, 3.8, 6.3};

std::vector<float> gs_ramp(int64_t n, float scale)
{
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i)
    {
        v[i] = scale * float((i * 7) % 11) - 2.0f;
    }
    return v;
}

}  // namespace

TT_INSTANTIATE_DEVICE_TESTS(TinyTorchGridSamplePadding);

// a sample far outside must not pick up weight from clamped border pixels with zero padding
TEST_P(TinyTorchGridSamplePadding, ZeroPaddingBilinear2d)
{
    Tensor input = tttest::leaf({1, 2, 3, 4}, {1, 1, 2, 2}, device());
    // pixel coords (x,y): (-3,0.5) far outside, (-0.5,-0.5) corner, (-0.5,0.5) left edge, (0.5,0.5) inside
    std::vector<float> g;
    for (auto [x, y] : std::vector<std::pair<double, double>>{{-3, 0.5}, {-0.5, -0.5}, {-0.5, 0.5}, {0.5, 0.5}})
    {
        g.push_back((float)gs_pixel_to_grid(x, 2, false));
        g.push_back((float)gs_pixel_to_grid(y, 2, false));
    }
    Tensor grid = tttest::make_tensor(g, {1, 1, 4, 2}, device());

    Tensor out = tinytorch::nn::functional::grid_sample(input, grid, gs_opts(kBilinear, kZero, false));
    TT_EXPECT_CLOSE(out, tttest::make_tensor({0, 0.25, 1.0, 2.5}, {1, 1, 1, 4}, device()), 1e-5, 1e-6);

    tinytorch::backward(tinytorch::sum(out));
    // input grad = sum of the in-bounds weights: corner 0.25, edge 0.25 + 0.25 (rows 0 and 1), center 0.25 each
    TT_EXPECT_CLOSE(input.grad(), tttest::make_tensor({0.75, 0.25, 0.5, 0.25}, {1, 1, 2, 2}, device()), 1e-5, 1e-6);

    Tensor out_border = tinytorch::nn::functional::grid_sample(input, grid, gs_opts(kBilinear, kBorder, false));
    TT_EXPECT_CLOSE(out_border, tttest::make_tensor({2, 1, 2, 2.5}, {1, 1, 1, 4}, device()), 1e-5, 1e-6);
}

TEST_P(TinyTorchGridSamplePadding, ZeroPaddingBilinear3d)
{
    std::vector<float> data(8);
    for (int i = 0; i < 8; ++i)
    {
        data[i] = i + 1;
    }
    Tensor input = tttest::leaf(data, {1, 1, 2, 2, 2}, device());
    // (x,y,z) pixel coords: x far outside, all axes inside -> center, one corner (-0.5,-0.5,-0.5)
    std::vector<float> g;
    for (auto p : std::vector<std::vector<double>>{{-3.5, 0.5, 0.5}, {0.5, 0.5, 0.5}, {-0.5, -0.5, -0.5}})
    {
        for (double x : p)
        {
            g.push_back((float)gs_pixel_to_grid(x, 2, false));
        }
    }
    Tensor grid = tttest::make_tensor(g, {1, 1, 1, 3, 3}, device());
    Tensor out  = tinytorch::nn::functional::grid_sample(input, grid, gs_opts(kBilinear, kZero, false));
    TT_EXPECT_CLOSE(out, tttest::make_tensor({0, 4.5, 0.125}, {1, 1, 1, 1, 3}, device()), 1e-5, 1e-6);

    tinytorch::backward(tinytorch::sum(out));
    TT_EXPECT_CLOSE(
        input.grad(),
        tttest::make_tensor({0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125}, {1, 1, 2, 2, 2}, device()), 1e-5,
        1e-6);
}

// nearest rounds half to even (like PyTorch) and returns 0 outside with zero padding
TEST_P(TinyTorchGridSamplePadding, Nearest)
{
    Tensor input = tttest::leaf({1, 2, 3, 4}, {1, 1, 1, 4}, device());
    Tensor grid  = gs_row_grid({0.5, 1.5, 2.5, -0.4, -0.6, 3.4, 3.6, -3}, false, device());

    Tensor out = tinytorch::nn::functional::grid_sample(input, grid, gs_opts(kNearest, kZero, false));
    TT_EXPECT_CLOSE(out, tttest::make_tensor({1, 3, 3, 1, 0, 4, 0, 0}, {1, 1, 1, 8}, device()), 0, 0);

    tinytorch::backward(tinytorch::sum(out));
    TT_EXPECT_CLOSE(input.grad(), tttest::make_tensor({2, 0, 2, 1}, {1, 1, 1, 4}, device()), 0, 0);

    Tensor out_border = tinytorch::nn::functional::grid_sample(input, grid, gs_opts(kNearest, kBorder, false));
    TT_EXPECT_CLOSE(out_border, tttest::make_tensor({1, 3, 3, 1, 1, 4, 4, 1}, {1, 1, 1, 8}, device()), 0, 0);

    std::vector<float> data(8);
    for (int i = 0; i < 8; ++i)
    {
        data[i] = i + 1;
    }
    Tensor input3 = tttest::leaf(data, {1, 1, 2, 2, 2}, device());
    Tensor grid3  = tttest::make_tensor({-3, 0, 0, 0.2, 0.2, 0.2}, {1, 1, 1, 2, 3}, device());
    Tensor out3   = tinytorch::nn::functional::grid_sample(input3, grid3, gs_opts(kNearest, kZero, false));
    TT_EXPECT_CLOSE(out3, tttest::make_tensor({0, 8}, {1, 1, 1, 1, 2}, device()), 0, 0);
    tinytorch::backward(tinytorch::sum(out3));
    TT_EXPECT_CLOSE(input3.grad(), tttest::make_tensor({0, 0, 0, 0, 0, 0, 0, 1}, {1, 1, 2, 2, 2}, device()), 0, 0);
}

TEST_P(TinyTorchGridSamplePadding, Reflection)
{
    Tensor input = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 1, 4}, device());

    // align_corners=false reflects at -0.5 and 3.5
    Tensor grid = gs_row_grid({-1.25, 4.25, 9.25, 1.5}, false, device());
    Tensor out  = tinytorch::nn::functional::grid_sample(input, grid, gs_opts(kBilinear, kReflect, false));
    TT_EXPECT_CLOSE(out, tttest::make_tensor({1.25, 3.75, 2.25, 2.5}, {1, 1, 1, 4}, device()), 1e-5, 1e-5);

    // align_corners=true reflects at 0 and 3
    Tensor grid_ac = gs_row_grid({-1.25, 3.5, -7.25}, true, device());
    Tensor out_ac  = tinytorch::nn::functional::grid_sample(input, grid_ac, gs_opts(kBilinear, kReflect, true));
    TT_EXPECT_CLOSE(out_ac, tttest::make_tensor({2.25, 3.5, 2.25}, {1, 1, 1, 3}, device()), 1e-5, 1e-5);

    Tensor out_n = tinytorch::nn::functional::grid_sample(input, grid, gs_opts(kNearest, kReflect, false));
    TT_EXPECT_CLOSE(out_n, tttest::make_tensor({1, 4, 2, 3}, {1, 1, 1, 4}, device()), 0, 0);
}

// forward/backward consistency (input and grid gradient) for every mode/padding/align_corners combination
TEST_P(TinyTorchGridSamplePadding, GradientAllModes2d)
{
    const int64_t H = 3, W = 4;
    const auto& xs  = gs_check_coords;
    const int64_t n = (int64_t)xs.size();
    for (auto mode : {kBilinear, kNearest})
    {
        for (auto padding : {kZero, kBorder, kReflect})
        {
            for (bool ac : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "mode " << mode << " padding " << padding << " ac " << ac);
                std::vector<float> g;
                for (int64_t i = 0; i < n; ++i)
                {
                    g.push_back((float)gs_pixel_to_grid(xs[i], W, ac));
                    g.push_back((float)gs_pixel_to_grid(xs[(i * 4 + 3) % n], H, ac));
                }
                Tensor input  = tttest::leaf(gs_ramp(2 * H * W, 0.5f), {1, 2, H, W}, device());
                Tensor grid   = tttest::leaf(g, {1, 1, n, 2}, device());
                Tensor weight = tttest::make_tensor(gs_ramp(2 * n, 0.3f), {1, 2, 1, n}, device());
                auto opts     = gs_opts(mode, padding, ac);
                tttest::check_grads(
                    {input, grid}, [&](const std::vector<Tensor>& p)
                    { return tinytorch::sum(tinytorch::nn::functional::grid_sample(p[0], p[1], opts) * weight); });
            }
        }
    }
}

TEST_P(TinyTorchGridSamplePadding, GradientAllModes3d)
{
    const int64_t D = 2, H = 3, W = 4;
    const auto& xs  = gs_check_coords;
    const int64_t n = (int64_t)xs.size();
    for (auto mode : {kBilinear, kNearest})
    {
        for (auto padding : {kZero, kBorder, kReflect})
        {
            for (bool ac : {false, true})
            {
                SCOPED_TRACE(::testing::Message() << "mode " << mode << " padding " << padding << " ac " << ac);
                std::vector<float> g;
                for (int64_t i = 0; i < n; ++i)
                {
                    g.push_back((float)gs_pixel_to_grid(xs[i], W, ac));
                    g.push_back((float)gs_pixel_to_grid(xs[(i * 4 + 3) % n], H, ac));
                    g.push_back((float)gs_pixel_to_grid(xs[(i * 2 + 5) % n], D, ac));
                }
                Tensor input  = tttest::leaf(gs_ramp(2 * D * H * W, 0.5f), {1, 2, D, H, W}, device());
                Tensor grid   = tttest::leaf(g, {1, 1, 1, n, 3}, device());
                Tensor weight = tttest::make_tensor(gs_ramp(2 * n, 0.3f), {1, 2, 1, 1, n}, device());
                auto opts     = gs_opts(mode, padding, ac);
                tttest::check_grads(
                    {input, grid}, [&](const std::vector<Tensor>& p)
                    { return tinytorch::sum(tinytorch::nn::functional::grid_sample(p[0], p[1], opts) * weight); });
            }
        }
    }
}

// half/double run on both backends (2D and 3D, forward and backward) and agree with float
TEST_P(TinyTorchGridSamplePadding, Dtypes)
{
    for (int dims : {2, 3})
    {
        SizeType in_size           = dims == 2 ? SizeType{1, 2, 3, 4} : SizeType{1, 2, 2, 3, 4};
        SizeType grid_size         = dims == 2 ? SizeType{1, 1, 5, 2} : SizeType{1, 1, 1, 5, 3};
        std::vector<float> in_data = gs_ramp(in_size.numel(), 0.25f);
        std::vector<float> g;
        for (int64_t i = 0; i < grid_size.numel(); ++i)
        {
            g.push_back(-1.3f + 0.17f * float(i % 13));
        }

        Tensor ref_in   = tttest::leaf(in_data, in_size, device());
        Tensor ref_grid = tttest::leaf(g, grid_size, device());
        auto opts       = gs_opts(kBilinear, kZero, false);
        Tensor ref_out  = tinytorch::nn::functional::grid_sample(ref_in, ref_grid, opts);
        tinytorch::backward(tinytorch::sum(ref_out));

        for (auto dtype : {tinytorch::kDouble, tinytorch::kHalf})
        {
            SCOPED_TRACE(::testing::Message() << "dims " << dims << " dtype " << dtype);
            double tol = dtype == tinytorch::kHalf ? 2e-2 : 1e-5;
            Tensor in  = tttest::make_tensor(in_data, in_size, device(), dtype);
            Tensor gr  = tttest::make_tensor(g, grid_size, device(), dtype);
            in.set_requires_grad(true, true);
            gr.set_requires_grad(true, true);
            Tensor out = tinytorch::nn::functional::grid_sample(in, gr, opts);
            EXPECT_EQ(out.scalar_type(), dtype);
            TT_EXPECT_CLOSE(out.to(kFloat), ref_out, tol, tol);
            tinytorch::backward(tinytorch::sum(out));
            TT_EXPECT_CLOSE(in.grad().to(kFloat), ref_in.grad(), tol, tol);
            TT_EXPECT_CLOSE(gr.grad().to(kFloat), ref_grid.grad(), tol, 5 * tol);
        }
    }
}
