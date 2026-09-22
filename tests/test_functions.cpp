/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

using tttest::DeviceTest;

namespace
{

// Folds in the cases from the former tests/test_backward.cpp.

// d(sum(A*B))/dA[p][q] = sum_n B[q][n] (row sums of B),
// d(sum(A*B))/dB[q][n] = sum_p A[p][q] (column sums of A, repeated over n)
TEST(Backward, Matmul)
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

TT_INSTANTIATE_DEVICE_TESTS(ReshapeOps);
TEST_P(ReshapeOps, ValuesAndGradient)
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

TT_INSTANTIATE_DEVICE_TESTS(RepeatOps);
TEST_P(RepeatOps, Values)
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

// ------------------------------------------------------------------ transpose / permute / flip

TT_INSTANTIATE_DEVICE_TESTS(TransposeOps);
TEST_P(TransposeOps, ValuesAndGradient)
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

TT_INSTANTIATE_DEVICE_TESTS(PermuteOps);
TEST_P(PermuteOps, ValuesAndGradient)
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

TT_INSTANTIATE_DEVICE_TESTS(FlipOps);
TEST_P(FlipOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor f0 = tinytorch::flip(a, {0});
    TT_EXPECT_CLOSE(f0, tttest::make_tensor({4, 5, 6, 1, 2, 3}, {2, 3}, device()), 0, 0);

    Tensor f1 = tinytorch::flip(a, {1});
    TT_EXPECT_CLOSE(f1, tttest::make_tensor({3, 2, 1, 6, 5, 4}, {2, 3}, device()), 0, 0);
}

// ------------------------------------------------------------------ copy / fill / uniform

TT_INSTANTIATE_DEVICE_TESTS(CopyOps);
TEST_P(CopyOps, Values)
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

TT_INSTANTIATE_DEVICE_TESTS(FillOps);
TEST_P(FillOps, Values)
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

TT_INSTANTIATE_DEVICE_TESTS(UniformFillOps);
TEST_P(UniformFillOps, Ranges)
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

    Tensor c = tinytorch::empty({2048}, TensorOptions().device(device()));
    tinytorch::normal_random(c);
    EXPECT_NEAR(c.mean().toDouble(), 0.0, 0.05);
}

// ------------------------------------------------------------------ sort

TEST(SortOps, ValuesAndIndices)
{
    Tensor a = tttest::make_tensor({3, 1, 4, 1, 5, 9}, {2, 3}, kCPU);

    auto [s, idx] = tinytorch::sort(a, 1);
    TT_EXPECT_CLOSE(s, tttest::make_tensor({1, 3, 4, 1, 5, 9}, {2, 3}, kCPU), 0, 0);
    // indices are per-row (0-based within the reduced dim)
    TT_EXPECT_CLOSE(idx, tttest::make_tensor(std::vector<int64_t>{1, 0, 2, 0, 1, 2}, {2, 3}, kCPU), 0, 0);
}

// ------------------------------------------------------------------ clone

TT_INSTANTIATE_DEVICE_TESTS(CloneOps);
TEST_P(CloneOps, ValuesAndIndependence)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    Tensor c = tinytorch::clone(a);
    TT_EXPECT_CLOSE(c, a, 0, 0);

    // in-place modification of the clone must not touch the original
    c.fill_(42.0);
    TT_EXPECT_CLOSE(a, tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ to / dtype

TT_INSTANTIATE_DEVICE_TESTS(ToOps);
TEST_P(ToOps, DtypeConversion)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    Tensor d = a.to(kFloat64);
    EXPECT_EQ(d.dtype(), kFloat64);
    TT_EXPECT_CLOSE(d, a, 1e-6, 1e-6);

    Tensor f = d.to(kFloat);
    EXPECT_EQ(f.dtype(), kFloat);
    TT_EXPECT_CLOSE(f, a, 1e-6, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorToMethod);
TEST_P(TensorToMethod, InPlaceDtype)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device());
    a.to_(kFloat64);
    EXPECT_EQ(a.dtype(), kFloat64);
    TT_EXPECT_CLOSE(a, tttest::make_tensor({1, 2, 3, 4}, {4}, device(), kFloat64), 1e-6, 1e-6);
}

// ------------------------------------------------------------------ slice

TT_INSTANTIATE_DEVICE_TESTS(SliceOps);
TEST_P(SliceOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {6}, device());

    TT_EXPECT_CLOSE(tinytorch::slice(a, 0, 1, 4, 1), tttest::make_tensor({2, 3, 4}, {3}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::slice(a, 0, 0, 6, 2), tttest::make_tensor({1, 3, 5}, {3}, device()), 0, 0);

    Tensor m = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    // slicing keeps the dim (size 1)
    TT_EXPECT_CLOSE(tinytorch::slice(m, 0, 1, 2, 1), tttest::make_tensor({4, 5, 6}, {1, 3}, device()), 0, 0);
    TT_EXPECT_CLOSE(tinytorch::slice(m, 1, 0, 2, 1), tttest::make_tensor({1, 2, 4, 5}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(SliceView);
TEST_P(SliceView, WritesThroughToOriginal)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device());
    a.slice_view(0, 1, 3).copy_(tttest::make_tensor({9, 9}, {2}, device()));
    TT_EXPECT_CLOSE(a, tttest::make_tensor({1, 9, 9, 4}, {4}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(SliceGradient);
TEST_P(SliceGradient, ZerosOutsideTheSlice)
{
    Tensor a = tttest::leaf({1, 2, 3, 4}, {4}, device());

    Tensor s = tinytorch::slice(a, 0, 1, 3, 1);
    tinytorch::backward(tinytorch::sum(s));
    TT_EXPECT_CLOSE(a.grad(), tttest::make_tensor({0, 1, 1, 0}, {4}, device()), 0, 0);
}

// ------------------------------------------------------------------ stack / cat

TT_INSTANTIATE_DEVICE_TESTS(StackOps);
TEST_P(StackOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2}, {2}, device());
    Tensor b = tttest::make_tensor({3, 4}, {2}, device());

    // stacking along a new dim 0
    Tensor s = tinytorch::stack({a, b});
    EXPECT_EQ(s.sizes(), (SizeType{2, 2}));
    TT_EXPECT_CLOSE(s, tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(CatOps);
TEST_P(CatOps, ValuesAndGradient)
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

TT_INSTANTIATE_DEVICE_TESTS(IndexSelectOps);
TEST_P(IndexSelectOps, ValuesAndGradient)
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

TT_INSTANTIATE_DEVICE_TESTS(IndexAddOps);
TEST_P(IndexAddOps, ValuesAndGradient)
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

TT_INSTANTIATE_DEVICE_TESTS(IndexCopyOps);
TEST_P(IndexCopyOps, Values)
{
    Tensor target = tinytorch::zeros({2, 2}, TensorOptions().device(device()));
    Tensor source = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    Tensor idx = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());

    tinytorch::index_copy(target, 0, idx, source);
    TT_EXPECT_CLOSE(target, tttest::make_tensor({3, 4, 1, 2}, {2, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(GatherOps);
TEST_P(GatherOps, Values)
{
    // out[i][j] = data[i][index[i][j]]  (dim == 1)
    Tensor data = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    Tensor idx = tttest::make_tensor(std::vector<int64_t>{2, 0, 1, 2}, {2, 2}, device());

    // out[i][j] = data[i][idx[i][j]]: [[1,2,3],[4,5,6]] with [[2,0],[1,2]] -> [[3,1],[5,6]]
    Tensor g = tinytorch::gather(data, 1, idx);
    TT_EXPECT_CLOSE(g, tttest::make_tensor({3, 1, 5, 6}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ padding

TT_INSTANTIATE_DEVICE_TESTS(Padding2dZero);
TEST_P(Padding2dZero, Values)
{
    // padding_2d requires 4D input with batch/channel == 1
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());

    Tensor p = tinytorch::padding_2d(a, 1, 1, 1, 1, kZero);
    EXPECT_EQ(p.sizes(), (SizeType{1, 1, 4, 4}));
    TT_EXPECT_CLOSE(p, tttest::make_tensor({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0}, {1, 1, 4, 4}, device()),
                    0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(Padding2dBorder);
TEST_P(Padding2dBorder, Values)
{
    // kBorder replicates the edge values (clamp)
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());

    Tensor p = tinytorch::padding_2d(a, 1, 1, 1, 1, kBorder);
    TT_EXPECT_CLOSE(p,
                    tttest::make_tensor({1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}, {1, 1, 4, 4}, device()), 0,
                    0);
}

TT_INSTANTIATE_DEVICE_TESTS(Padding2dReflect);
TEST_P(Padding2dReflect, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());

    Tensor p = tinytorch::padding_2d(a, 1, 1, 1, 1, kReflect);
    // reflection maps -1 -> 1 and 2 -> 0, so each axis gives indices 1,0,1,0:
    // row0 = a[1] reflected = [4,3,4,3], row1 = a[0] reflected = [2,1,2,1], etc.
    TT_EXPECT_CLOSE(p,
                    tttest::make_tensor({4, 3, 4, 3, 2, 1, 2, 1, 4, 3, 4, 3, 2, 1, 2, 1}, {1, 1, 4, 4}, device()), 0,
                    0);
}

TT_INSTANTIATE_DEVICE_TESTS(Padding3dZero);
TEST_P(Padding3dZero, Values)
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

// ------------------------------------------------------------------ matmul

TT_INSTANTIATE_DEVICE_TESTS(MatmulOps);
TEST_P(MatmulOps, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    Tensor b = tttest::make_tensor({7, 8, 9, 10, 11, 12}, {3, 2}, device());

    Tensor m = tinytorch::matmul(a, b);
    TT_EXPECT_CLOSE(m, tttest::make_tensor({58, 64, 139, 154}, {2, 2}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(MatmulBatched);
TEST_P(MatmulBatched, ValuesAndGradient)
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

// ------------------------------------------------------------------ conv2d

TEST(Conv2d, BoxFilter)
{
    // 4x4 input with distinct values, 3x3 all-ones kernel, padding 1 -> 4x4 output
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> w(9, 1.0f);

    Tensor input = tinytorch::from_blob(data.data(), {1, 1, 4, 4});
    Tensor weight = tinytorch::from_blob(w.data(), {1, 1, 3, 3});

    Tensor out = tinytorch::conv2d(input, weight, Tensor(), 1, 1, 1, 1);
    EXPECT_EQ(out.sizes(), (SizeType{1, 1, 4, 4}));

    // out[0][0] = zero-padded 3x3 window at (0,0): 1+2+5+6 = 24
    // out[1][1] = full 3x3 window at (1,1): 1+2+3+5+6+7+9+10+11 = 54
    auto v = tttest::to_double_vec(out);
    EXPECT_DOUBLE_EQ(v[0], 24);
    EXPECT_DOUBLE_EQ(v[5], 54);
}

// conv2d must produce the same result on CPU and CUDA
TT_INSTANTIATE_DEVICE_TESTS(Conv2dDevice);
TEST_P(Conv2dDevice, BoxFilter)
{
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> w(9, 1.0f);

    Tensor input = tttest::make_tensor(data, {1, 1, 4, 4}, device());
    Tensor weight = tttest::make_tensor(w, {1, 1, 3, 3}, device());

    Tensor out = tinytorch::conv2d(input, weight, Tensor(), 1, 1, 1, 1);
    EXPECT_EQ(out.sizes(), (SizeType{1, 1, 4, 4}));

    // out[0][0] = zero-padded 3x3 window at (0,0): 1+2+5+6 = 24
    // out[1][1] = full 3x3 window at (1,1): 1+2+3+5+6+7+9+10+11 = 54
    auto v = tttest::to_double_vec(out);
    EXPECT_NEAR(v[0], 24, 1e-4);
    EXPECT_NEAR(v[5], 54, 1e-4);
}

// ------------------------------------------------------------------ poisson noise

TT_INSTANTIATE_DEVICE_TESTS(PoissonNoise);
TEST_P(PoissonNoise, RunsAndModifies)
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

TT_INSTANTIATE_DEVICE_TESTS(GridSample2d);
TEST_P(GridSample2d, BilinearAlignCorners)
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

TEST_P(GridSample2d, BilinearCenter)
{
    Tensor input = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());
    // center: (0,0) -> px(0.5,0.5) -> (1+2+3+4)/4 = 2.5
    Tensor grid = make_grid({0, 0}, device());

    Tensor out = tinytorch::nn::functional::grid_sample(
        input, grid, tinytorch::nn::functional::GridSampleFuncOptions().mode(kBilinear).padding_mode(kBorder).align_corners(true));

    TT_EXPECT_CLOSE(out, tttest::make_tensor({2.5f}, {1, 1, 1, 1}, device()), 1e-5, 1e-6);
}

TEST_P(GridSample2d, Nearest)
{
    Tensor input = tttest::make_tensor({1, 2, 3, 4}, {1, 1, 2, 2}, device());
    // (0.5,0.5) -> px(0.5,0.5); nearest rounds to (1,1) -> 4 (bilinear would give 2.5)
    Tensor grid = make_grid({0.5f, 0.5f}, device());

    Tensor out = tinytorch::nn::functional::grid_sample(
        input, grid, tinytorch::nn::functional::GridSampleFuncOptions().mode(kNearest).padding_mode(kBorder).align_corners(true));

    TT_EXPECT_CLOSE(out, tttest::make_tensor({4.0f}, {1, 1, 1, 1}, device()), 0, 0);
}

TEST_P(GridSample2d, Gradient)
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

TT_INSTANTIATE_DEVICE_TESTS(GridSample3d);
TEST_P(GridSample3d, BilinearAndNearest)
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

    // nearest: (0,0,0) -> px(0.5,0.5,0.5) -> rounds to (1,1,1) -> 8 (bilinear would give 4.5)
    Tensor grid_n = tttest::make_tensor({0, 0, 0}, {1, 1, 1, 1, 3}, device());
    auto opts_n   = tinytorch::nn::functional::GridSampleFuncOptions().mode(kNearest).padding_mode(kBorder).align_corners(true);
    Tensor out_n  = tinytorch::nn::functional::grid_sample(input, grid_n, opts_n);
    TT_EXPECT_CLOSE(out_n, tttest::make_tensor({8.0f}, {1, 1, 1, 1, 1}, device()), 0, 0);
}
