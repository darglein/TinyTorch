/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_tensor_creation.h"
#include "torch/core/ops/ops_impl.h"



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
    return empty_like(t, t.options().requires_grad(false));
}

Tensor empty_like(Tensor t, TensorOptions options)
{
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
    return full(t.sizes(), value, options.requires_grad(false));
}

Tensor full_like(Tensor t, float value)
{
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
    return zeros(t.sizes(), options.requires_grad(false));
}

Tensor zeros_like(Tensor t)
{
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
    return ones(t.sizes(), options.requires_grad(false));
}

Tensor ones_like(Tensor t)
{
    return ones_like(t, t.options().requires_grad(false));
}


// ================================================================================
// Rand Tensor Creation

static int64_t& current_seed()
{
    static thread_local int64_t s = 9036515235;
    return s;
}


void manual_seed(int64_t seed)
{
    current_seed() = seed;
}

int64_t get_seed()
{
    return current_seed();
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
    auto storage = std::make_shared<StorageImpl>(data, size, options.device_);

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

Tensor range(double start, double end, TensorOptions options, double step)
{
    int64_t count = int64_t((end - start) / step) + 1;
    Tensor t      = empty({count}, options);
    SELECT_DEVICE(t.device(), range_impl,t,start,end,step);
    return t;
}
Tensor range(double start, double end, double step)
{
    return range(start, end, TensorOptions(), step);
}



}  // namespace tinytorch