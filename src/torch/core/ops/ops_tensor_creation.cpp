/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_tensor_creation.h"

#include <chrono>

#include "torch/core/ops/ops_impl.h"
#include "torch/core/tensor_impl.h"


namespace tinytorch
{



// ================================================================================
// Empty Tensor Creation


Tensor empty(const SizeType& sizes, TensorOptions options)
{
    if (!GradMode::is_enabled())
    {
        options.requires_grad_ = false;
    }
    Tensor t(TensorImpl::create(sizes, options));
    // for debugging
    // fill(t, -128356);
    t.set_requires_grad(options.requires_grad_);
    return t;
}

Tensor empty_like(Tensor t)
{
    if (!t.defined())
    {
        return {};
    }
    return empty_like(t, t.options().requires_grad(false));
}

Tensor empty_like(Tensor t, TensorOptions options)
{
    if (!t.defined())
    {
        return {};
    }
    return empty(t.sizes(), options.requires_grad(false));
}


// ================================================================================
// Full Tensor Creation

Tensor full(const SizeType& sizes, float value, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    NoGradGuard ngg;
    fill(t, value);
    return t;
}
Tensor full_like(Tensor t, float value, TensorOptions options)
{
    if (!t.defined())
    {
        return {};
    }
    return full(t.sizes(), value, options.requires_grad(false));
}

Tensor full_like(Tensor t, float value)
{
    if (!t.defined())
    {
        return {};
    }
    return full_like(t, value, t.options().requires_grad(false));
}


// ================================================================================
// Zero Tensor Creation

Tensor zeros(const SizeType& sizes, TensorOptions options)
{
    return full(sizes, 0, options);
}
Tensor zeros_like(Tensor t, TensorOptions options)
{
    if (!t.defined())
    {
        return {};
    }
    return zeros(t.sizes(), options.requires_grad(false));
}

Tensor zeros_like(Tensor t)
{
    if (!t.defined())
    {
        return {};
    }
    return zeros_like(t, t.options().requires_grad(false));
}

// ================================================================================
// One Tensor Creation

Tensor ones(const SizeType& sizes, TensorOptions options)
{
    return full(sizes, 1, options);
}
Tensor ones_like(Tensor t, TensorOptions options)
{
    if (!t.defined())
    {
        return {};
    }
    return ones(t.sizes(), options.requires_grad(false));
}

Tensor ones_like(Tensor t)
{
    if (!t.defined())
    {
        return {};
    }
    return ones_like(t, t.options().requires_grad(false));
}


// ================================================================================
// Rand Tensor Creation

std::mt19937& generator()
{
    static thread_local std::mt19937 gen((uint32_t)std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    return gen;
}

void manual_seed(int64_t seed)
{
    generator().seed((uint32_t)seed);
}

Tensor rand(const SizeType& sizes, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    NoGradGuard ngg;
    uniform(t);
    return t;
}

Tensor randint(int low, int high, const SizeType& sizes, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    uniform_int(t, low, high);
    return t;
}

Tensor rand_like(Tensor t)
{
    return rand(t.sizes(), t.options().requires_grad(false));
}
Tensor from_blob(void* data, const SizeType& sizes, const SizeType& stride, TensorOptions options)
{
    CHECK(options.requires_grad_ == false);
    int64_t size = sizes.numel() * elementSize(options.dtype_);
    auto storage = std::make_shared<StorageImpl>(data, size, 0, options);

    Tensor t(TensorImpl::create(storage, 0, sizes, stride, options));
    return t;
}
Tensor from_blob(void* data, const SizeType& sizes, TensorOptions options)
{
    CHECK(options.requires_grad_ == false);
    return from_blob(data, sizes, CompactStrideForSize(sizes), options);
}
Tensor from_blob(void* data, const SizeType& sizes, ScalarType type)
{
    return from_blob(data, sizes, CompactStrideForSize(sizes), TensorOptions().dtype(type));
}
Tensor from_blob(void* data, const SizeType& sizes, const SizeType& stride, ScalarType type)
{
    return from_blob(data, sizes, stride, TensorOptions().dtype(type));
}

Tensor range(double start, double end, double step, TensorOptions options)
{
    int64_t count = int64_t((end - start) / step) + 1;
    Tensor t      = empty({count}, options);
    SELECT_DEVICE(t.device(), range_impl, t, start, end, step);
    return t;
}



}  // namespace tinytorch