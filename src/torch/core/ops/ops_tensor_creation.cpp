/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_tensor_creation.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>

#include "torch/core/ops/ops_impl.h"
#include "torch/core/tensor_impl.h"


namespace tinytorch
{



// ================================================================================
// Empty Tensor Creation


Tensor empty(const SizeType& sizes, TensorOptions options)
{
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

Tensor full(const SizeType& sizes, double value, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    NoGradGuard ngg;
    fill(t, value);
    return t;
}
Tensor full_like(Tensor t, double value, TensorOptions options)
{
    if (!t.defined())
    {
        return {};
    }
    return full(t.sizes(), value, options.requires_grad(false));
}

Tensor full_like(Tensor t, double value)
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

namespace
{
std::mutex seed_mutex;
uint64_t global_seed = 0;
std::atomic<uint64_t> global_seed_generation{0};
std::atomic<uint64_t> generator_thread_counter{0};

struct ThreadGenerator
{
    std::mt19937 gen;
    uint64_t thread_index = generator_thread_counter.fetch_add(1);
    uint64_t generation   = 0;
    // set by manual_seed() in this thread: manual_seed() calls of other threads don't re-seed it anymore
    bool seeded_by_this_thread = false;

    ThreadGenerator()
    {
        uint64_t t  = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        uint32_t r0 = 0, r1 = 0;
        try
        {
            std::random_device rd;
            r0 = rd();
            r1 = rd();
        }
        catch (...)
        {
        }
        std::seed_seq seq{r0, r1, uint32_t(t), uint32_t(t >> 32), uint32_t(thread_index), uint32_t(thread_index >> 32)};
        gen.seed(seq);
    }
};

ThreadGenerator& thread_generator()
{
    static thread_local ThreadGenerator g;
    return g;
}
}  // namespace

std::mt19937& generator()
{
    auto& g = thread_generator();
    if (!g.seeded_by_this_thread && global_seed_generation.load(std::memory_order_acquire) != g.generation)
    {
        // manual_seed() was called in another thread: derive this thread's stream from (seed, thread index)
        std::lock_guard<std::mutex> lock(seed_mutex);
        std::seed_seq seq{uint32_t(global_seed), uint32_t(global_seed >> 32), uint32_t(g.thread_index),
                          uint32_t(g.thread_index >> 32)};
        g.gen.seed(seq);
        g.generation = global_seed_generation.load(std::memory_order_relaxed);
    }
    return g.gen;
}

void manual_seed(int64_t seed)
{
    auto& g    = thread_generator();
    uint64_t s = (uint64_t)seed;
    if (s <= 0xFFFFFFFFull)
    {
        g.gen.seed((uint32_t)s);
    }
    else
    {
        std::seed_seq seq{uint32_t(s), uint32_t(s >> 32)};
        g.gen.seed(seq);
    }
    g.seeded_by_this_thread = true;

    std::lock_guard<std::mutex> lock(seed_mutex);
    global_seed = s;
    global_seed_generation.fetch_add(1, std::memory_order_release);
}

Tensor rand(const SizeType& sizes, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    NoGradGuard ngg;
    uniform(t);
    return t;
}

Tensor randint(int64_t low, int64_t high, const SizeType& sizes, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    uniform_int(t, low, high);
    return t;
}

Tensor rand_like(Tensor t)
{
    return rand(t.sizes(), t.options().requires_grad(false));
}

Tensor randn(const SizeType& sizes, TensorOptions options)
{
    Tensor t = empty(sizes, options);
    NoGradGuard ngg;
    normal_random(t);
    return t;
}


Tensor randn_like(Tensor t)
{
    return randn(t.sizes(), t.options().requires_grad(false));
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
    CHECK(step != 0 && std::isfinite(step)) << "invalid step " << step;
    // end is inclusive; a quotient within rounding error of an integer counts as that integer (range(0, 0.3, 0.1)
    // has 4 elements)
    double q         = (end - start) / step;
    double q_rounded = std::round(q);
    if (std::abs(q - q_rounded) <= 1e-9 * std::max(1.0, std::abs(q)))
    {
        q = q_rounded;
    }
    int64_t count = q < 0 ? 0 : int64_t(std::floor(q)) + 1;
    Tensor t      = empty({count}, options);
    SELECT_DEVICE(t.device(), range_impl, t, start, end, step);
    return t;
}



}  // namespace tinytorch