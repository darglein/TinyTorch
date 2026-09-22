/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "test_utils.h"

using tttest::DeviceTest;

// ------------------------------------------------------------------ basic accessors

TT_INSTANTIATE_DEVICE_TESTS(TensorAccessors);
TEST_P(TensorAccessors, Properties)
{
    Tensor t = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    EXPECT_TRUE(t.defined());
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
    EXPECT_EQ(t.stride(0), 3);
    EXPECT_EQ(t.stride(1), 1);
    EXPECT_EQ(t.element_size(), 4);
    EXPECT_TRUE(t.is_contiguous());
    // is_leaf() is only true for requires-grad leaves (AccumulateGrad edge)
    EXPECT_FALSE(t.is_leaf());
    EXPECT_TRUE(tttest::leaf({1, 2, 3}, {3}, device()).is_leaf());
    EXPECT_EQ(t.device(), device());
    EXPECT_EQ(t.is_cuda(), on_cuda());
    EXPECT_EQ(t.is_cpu(), !on_cuda());
}

// ------------------------------------------------------------------ view / reshape / squeeze

TT_INSTANTIATE_DEVICE_TESTS(TensorView);
TEST_P(TensorView, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor v = a.view({3, 2});
    TT_EXPECT_CLOSE(v, tttest::make_tensor({1, 2, 3, 4, 5, 6}, {3, 2}, device()), 0, 0);

    // a view aliases the data: writing through it modifies the original
    v.slice_view(0, 0, 1).copy_(tttest::make_tensor({9, 9}, {1, 2}, device()));
    EXPECT_FLOAT_EQ(tttest::to_double_vec(a)[0], 9.0);
    EXPECT_FLOAT_EQ(tttest::to_double_vec(a)[1], 9.0);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorSqueezeUnsqueeze);
TEST_P(TensorSqueezeUnsqueeze, Shapes)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {1, 2, 2}, device());

    Tensor s0 = a.squeeze(0);
    EXPECT_EQ(s0.sizes(), (SizeType{2, 2}));

    // negative dim: squeeze(-3) == squeeze(0)
    Tensor sn = a.squeeze(-3);
    EXPECT_EQ(sn.sizes(), (SizeType{2, 2}));

    Tensor u = a.squeeze(0).unsqueeze(0);
    EXPECT_EQ(u.sizes(), (SizeType{1, 2, 2}));

    Tensor sv = a.squeeze_view();
    EXPECT_EQ(sv.sizes(), (SizeType{2, 2}));
}

TT_INSTANTIATE_DEVICE_TESTS(TensorCollapseView);
TEST_P(TensorCollapseView, CollapsesUnitDims)
{
    // 2x1x1x3 with compact strides {3,3,3,1}
    std::vector<float> data(6);
    for (int i = 0; i < 6; ++i)
    {
        data[i] = i + 1;
    }
    Tensor a = tttest::make_tensor(data, {2, 1, 1, 3}, device());

    // collapse all dims (excludeDim -1): the unit dims merge into the neighbors
    auto [c, new_dim] = a.collapse_view(-1);
    EXPECT_EQ(c.sizes(), (SizeType{6}));
    EXPECT_EQ(new_dim, -1);
    TT_EXPECT_CLOSE(c, tttest::make_tensor(data, {6}, device()), 0, 0);

    // collapse excluding dim 1 (which has size 1): the tail 1x3 collapses to 3
    auto [c1, new_dim1] = a.collapse_view(1);
    EXPECT_EQ(new_dim1, 1);
    EXPECT_EQ(c1.sizes(), (SizeType{2, 1, 3}));
}

// ------------------------------------------------------------------ permute / transpose / flip

TT_INSTANTIATE_DEVICE_TESTS(TensorPermuteView);
TEST_P(TensorPermuteView, StridesAndValues)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    Tensor p = a.permute_view({1, 0});
    EXPECT_EQ(p.sizes(), (SizeType{3, 2}));
    EXPECT_EQ(p.stride(0), 1);
    EXPECT_EQ(p.stride(1), 3);
    EXPECT_FALSE(p.is_contiguous());
    TT_EXPECT_CLOSE(p, tttest::make_tensor({1, 4, 2, 5, 3, 6}, {3, 2}, device()), 0, 0);

    // contiguous() materializes the layout
    Tensor c = p.contiguous();
    EXPECT_TRUE(c.is_contiguous());
    TT_EXPECT_CLOSE(c, p, 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorTransposeMethod);
TEST_P(TensorTransposeMethod, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());
    TT_EXPECT_CLOSE(a.transpose(0, 1), tttest::make_tensor({1, 4, 2, 5, 3, 6}, {3, 2}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorFlipMethod);
TEST_P(TensorFlipMethod, Values)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    TT_EXPECT_CLOSE(a.flip({0}), tttest::make_tensor({3, 4, 1, 2}, {2, 2}, device()), 0, 0);
    TT_EXPECT_CLOSE(a.flip({1}), tttest::make_tensor({2, 1, 4, 3}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ reinterpret

TT_INSTANTIATE_DEVICE_TESTS(TensorReinterpretView);
TEST_P(TensorReinterpretView, DtypeRoundTrip)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device());

    Tensor i = a.reinterpret_view(kInt32);
    EXPECT_EQ(i.dtype(), kInt32);
    EXPECT_EQ(i.numel(), 4);

    // round-trip back to float restores the values
    Tensor f = i.reinterpret_view(kFloat);
    TT_EXPECT_CLOSE(f, a, 0, 0);
}

// ------------------------------------------------------------------ in-place ops

TT_INSTANTIATE_DEVICE_TESTS(TensorInPlace);
TEST_P(TensorInPlace, ZeroFillUniform)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {4}, device());

    a.zero_();
    TT_EXPECT_CLOSE(a, tttest::make_tensor({0, 0, 0, 0}, {4}, device()), 0, 0);

    a.fill_(7.0);
    TT_EXPECT_CLOSE(a, tttest::make_tensor({7, 7, 7, 7}, {4}, device()), 0, 0);

    a.uniform_(0.0, 1.0);
    for (double x : tttest::to_double_vec(a))
    {
        EXPECT_GE(x, 0.0);
        EXPECT_LT(x, 1.0);
    }
}

TT_INSTANTIATE_DEVICE_TESTS(TensorResize);
TEST_P(TensorResize, GrowsAndKeepsData)
{
    Tensor a = tttest::make_tensor({1, 2}, {2}, device());

    a.resize_({4});
    EXPECT_EQ(a.sizes(), (SizeType{4}));
    EXPECT_FLOAT_EQ(tttest::to_double_vec(a)[0], 1.0f);
    EXPECT_FLOAT_EQ(tttest::to_double_vec(a)[1], 2.0f);
}

// ------------------------------------------------------------------ math member methods

TT_INSTANTIATE_DEVICE_TESTS(TensorMathMethods);
TEST_P(TensorMathMethods, SumMeanMinMax)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());

    TT_EXPECT_CLOSE(a.sum(), tttest::make_tensor({10}, {1}, device()), 1e-5, 1e-6);
    TT_EXPECT_CLOSE(a.sum(0, true), tttest::make_tensor({4, 6}, {1, 2}, device()), 1e-5, 1e-6);
    TT_EXPECT_CLOSE(a.mean(), tttest::make_tensor({2.5}, {1}, device()), 1e-5, 1e-6);

    auto [mn, mni] = a.min(1, false);
    TT_EXPECT_CLOSE(mn, tttest::make_tensor({1, 3}, {2}, device()), 0, 0);
    auto [mx, mxi] = a.max(1, false);
    TT_EXPECT_CLOSE(mx, tttest::make_tensor({2, 4}, {2}, device()), 0, 0);

    TT_EXPECT_CLOSE(a.abs(), tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device()), 0, 0);
    TT_EXPECT_CLOSE(a.square(), tttest::make_tensor({1, 4, 9, 16}, {2, 2}, device()), 0, 0);
    TT_EXPECT_CLOSE(a.sqrt(), tttest::make_tensor({1, 1.41421356f, 1.7320508f, 2}, {2, 2}, device()), 1e-5, 1e-5);

    TT_EXPECT_CLOSE(a.prod(1, false), tttest::make_tensor({2, 12}, {2}, device()), 1e-5, 1e-6);
    TT_EXPECT_CLOSE(a.cumsum(1), tttest::make_tensor({1, 3, 3, 7}, {2, 2}, device()), 1e-5, 1e-6);
    TT_EXPECT_CLOSE(a.cumprod(1), tttest::make_tensor({1, 2, 3, 12}, {2, 2}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorNormMethod);
TEST_P(TensorNormMethod, Frobenius)
{
    Tensor a = tttest::make_tensor({3, 4, 5, 12}, {2, 2}, device());
    TT_EXPECT_CLOSE(a.norm(2, 1, true), tttest::make_tensor({5, 13}, {2, 1}, device()), 1e-5, 1e-6);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorClampMethods);
TEST_P(TensorClampMethods, Values)
{
    Tensor a = tttest::make_tensor({-5, 0, 5}, {3}, device());

    Tensor c = a.clamp(-2, 2);
    TT_EXPECT_CLOSE(c, tttest::make_tensor({-2, 0, 2}, {3}, device()), 0, 0);

    a.clamp_(-1, 1);
    TT_EXPECT_CLOSE(a, tttest::make_tensor({-1, 0, 1}, {3}, device()), 0, 0);

    Tensor m = a.clamp_min(0);
    TT_EXPECT_CLOSE(m, tttest::make_tensor({0, 0, 1}, {3}, device()), 0, 0);

    m.clamp_max_(0.5);
    TT_EXPECT_CLOSE(m, tttest::make_tensor({0, 0, 0.5}, {3}, device()), 0, 0);
}

// ------------------------------------------------------------------ indexing member methods

TT_INSTANTIATE_DEVICE_TESTS(TensorIndexMethods);
TEST_P(TensorIndexMethods, SliceAndIndexSelect)
{
    Tensor a = tttest::make_tensor({1, 2, 3, 4, 5, 6}, {2, 3}, device());

    // slice keeps the dim (size 1); operator[] squeezes it
    TT_EXPECT_CLOSE(a.slice(0, 1, 2), tttest::make_tensor({4, 5, 6}, {1, 3}, device()), 0, 0);
    TT_EXPECT_CLOSE(a[1], tttest::make_tensor({4, 5, 6}, {3}, device()), 0, 0);

    Tensor idx = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());
    TT_EXPECT_CLOSE(a.index_select(0, idx), tttest::make_tensor({4, 5, 6, 1, 2, 3}, {2, 3}, device()), 0, 0);

    // gather along dim 1
    Tensor gidx = tttest::make_tensor(std::vector<int64_t>{2, 0}, {1, 2}, device());
    TT_EXPECT_CLOSE(a.gather(1, gidx), tttest::make_tensor({3, 1}, {1, 2}, device()), 0, 0);

    // index_add
    Tensor ia = a.index_add(0, idx, a);
    TT_EXPECT_CLOSE(ia, tttest::make_tensor({5, 7, 9, 5, 7, 9}, {2, 3}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorIndexCopyMethod);
TEST_P(TensorIndexCopyMethod, Values)
{
    Tensor target = tinytorch::zeros({2, 2}, TensorOptions().device(device()));
    Tensor source = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    Tensor idx = tttest::make_tensor(std::vector<int64_t>{1, 0}, {2}, device());

    target.index_copy_(0, idx, source);
    TT_EXPECT_CLOSE(target, tttest::make_tensor({3, 4, 1, 2}, {2, 2}, device()), 0, 0);
}

// ------------------------------------------------------------------ scalar extraction

TT_INSTANTIATE_DEVICE_TESTS(TensorItem);
TEST_P(TensorItem, Values)
{
    Tensor a = tttest::make_tensor({42}, {1}, device());

    EXPECT_DOUBLE_EQ(a.toDouble(), 42.0);
    EXPECT_FLOAT_EQ(a.toFloat(), 42.0f);
    EXPECT_EQ(a.toInt(), 42);
    EXPECT_EQ(a.toLong(), 42);

}

// ------------------------------------------------------------------ allclose

TT_INSTANTIATE_DEVICE_TESTS(TensorAllClose);
TEST_P(TensorAllClose, Comparison)
{
    Tensor a = tttest::make_tensor({1, 2, 3}, {3}, device());
    Tensor b = tttest::make_tensor({1, 2, 3}, {3}, device());
    Tensor c = tttest::make_tensor({1, 2, 4}, {3}, device());

    // Note: allclose() compares max|diff| <= atol only (rtol is accepted but unused)
    EXPECT_TRUE(a.allclose(b));
    EXPECT_TRUE(a.allclose(c, 0.0, 1.0));   // |3-4| = 1 <= atol 1
    EXPECT_FALSE(a.allclose(c, 0.0, 0.5));  // |3-4| = 1 > atol 0.5
}

// ------------------------------------------------------------------ autograd helpers

TT_INSTANTIATE_DEVICE_TESTS(TensorDetach);
TEST_P(TensorDetach, BreaksTheGraph)
{
    Tensor a = tttest::leaf({2.0f}, {1}, device());
    Tensor b = tttest::leaf({3.0f}, {1}, device());

    // detach removes the node edge
    Tensor d = a.detach();
    EXPECT_EQ(d.requires_grad(), false);

    // the detached tensor is not part of the graph: gradients only reach b
    Tensor r = d * 2 + b * 3;
    r.backward();
    EXPECT_FLOAT_EQ(b.grad().toFloat(), 3.0f);
    EXPECT_FLOAT_EQ(a.grad().toFloat(), 0.0f);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorRepeatMethods);
TEST_P(TensorRepeatMethods, Values)
{
    // repeat takes one count per dimension
    Tensor a = tttest::make_tensor({1, 2, 3, 4}, {2, 2}, device());
    TT_EXPECT_CLOSE(a.repeat({1, 2}), tttest::make_tensor({1, 2, 1, 2, 3, 4, 3, 4}, {2, 4}, device()), 0, 0);

    Tensor b = tttest::make_tensor({1, 2}, {2}, device());
    TT_EXPECT_CLOSE(b.repeat_interleave(2), tttest::make_tensor({1, 1, 2, 2}, {4}, device()), 0, 0);
}

TT_INSTANTIATE_DEVICE_TESTS(TensorSetData);
TEST_P(TensorSetData, Overwrites)
{
    Tensor a = tttest::make_tensor({1, 2}, {2}, device());
    Tensor b = tttest::make_tensor({7, 8}, {2}, device());

    a.set_data(b);
    EXPECT_EQ(a.sizes(), (SizeType{2}));
    TT_EXPECT_CLOSE(a, b, 0, 0);
}
