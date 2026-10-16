#pragma once

#include <cstdint>
#include <cstring>

#include "torch/tiny_torch_config.h"
#include <type_traits>

namespace tinytorch
{




struct Half
{
    uint16_t h = 0;

    struct FromBitsTag
    {
    };

    Half() {}

    // IEEE 754 binary32 -> binary16 with round-to-nearest-even, inf/NaN and subnormal support
    Half(float f) : h(float_to_bits(f)) {}
    Half(double d) : Half(float(d)) {}

    // integers are converted by value; use from_bits() to reinterpret raw binary16 bits
    template <typename I, typename std::enable_if<std::is_integral<I>::value, int>::type = 0>
    Half(I i) : Half(float(i))
    {
    }

    Half(uint16_t bits, FromBitsTag) : h(bits) {}
    static Half from_bits(uint16_t bits) { return Half(bits, FromBitsTag()); }

    operator float() const { return bits_to_float(h); }

    // Branch-free (selects only) so that loops converting Half vectorize.
    // https://gist.github.com/rygorous/2156668 (float_to_half_fast3_rtne / half_to_float_fast5, public domain)
    static uint16_t float_to_bits(float f)
    {
        constexpr uint32_t f32infty       = 255u << 23;
        constexpr uint32_t f16max         = (127u + 16u) << 23;
        constexpr uint32_t denorm_magic_u = ((127u - 15u) + (23u - 10u) + 1u) << 23;

        uint32_t x;
        std::memcpy(&x, &f, sizeof(x));
        uint32_t sign = x & 0x80000000u;
        uint32_t a    = x ^ sign;

        // half subnormal (or zero): the float addition rounds the mantissa (round-to-nearest-even)
        float af, denorm_magic;
        std::memcpy(&af, &a, sizeof(af));
        std::memcpy(&denorm_magic, &denorm_magic_u, sizeof(denorm_magic));
        float sub_f = af + denorm_magic;
        uint32_t sub;
        std::memcpy(&sub, &sub_f, sizeof(sub));
        sub -= denorm_magic_u;

        // normal: rebias the exponent and round-to-nearest-even (a carry into the exponent is correct, also to inf)
        uint32_t mant_odd = (a >> 13) & 1u;
        uint32_t normal   = (a + ((15u - 127u) << 23) + 0xfffu + mant_odd) >> 13;

        uint32_t inf_nan = a > f32infty ? 0x7e00u : 0x7c00u;
        uint32_t r       = a >= f16max ? inf_nan : (a < (113u << 23) ? sub : normal);
        return uint16_t(r | (sign >> 16));
    }

    static float bits_to_float(uint16_t bits)
    {
        constexpr uint32_t shifted_exp = 0x7c00u << 13;
        constexpr uint32_t magic_u     = 113u << 23;

        uint32_t o   = uint32_t(bits & 0x7fffu) << 13;
        uint32_t exp = shifted_exp & o;
        o += (127u - 15u) << 23;

        // inf/NaN: the exponent is all ones
        uint32_t o_normal = o + (exp == shifted_exp ? ((128u - 16u) << 23) : 0u);

        // zero/subnormal: renormalize with a float subtraction
        uint32_t o_sub_bits = o + (1u << 23);
        float o_sub, magic;
        std::memcpy(&o_sub, &o_sub_bits, sizeof(o_sub));
        std::memcpy(&magic, &magic_u, sizeof(magic));
        o_sub -= magic;
        std::memcpy(&o_sub_bits, &o_sub, sizeof(o_sub_bits));

        uint32_t result = (exp == 0 ? o_sub_bits : o_normal) | (uint32_t(bits & 0x8000u) << 16);
        float f;
        std::memcpy(&f, &result, sizeof(f));
        return f;
    }
};

template <typename T>
struct CpuComputeFloatType
{
    using Type = T;
};
template <>
struct CpuComputeFloatType<Half>
{
    using Type = float;
};

}  // namespace tinytorch


#if defined(TT_HAS_CUDA) && defined(__CUDACC__)
#    include <cuda_fp16.h>
namespace tinytorch
{
template <>
struct CpuComputeFloatType<__half>
{
    using Type = float;
};
}  // namespace tinytorch
inline __device__ __half round(__half a)
{
    return round(float(a));
}
inline __device__ __half sqrt(__half a)
{
    return hsqrt(a);
}
inline __device__ __half abs(__half a)
{
    return abs(float(a));
}
inline __device__ __half exp(__half a)
{
    return hexp(a);
}

inline __device__ __half log(__half a)
{
    return hlog(a);
}

inline __device__ __half log1p(__half a)
{
    return hlog(__half(1.f) + a);
}

inline __device__ __half sin(__half a)
{
    return hsin(a);
}

inline __device__ __half cos(__half a)
{
    return hcos(a);
}

inline __device__ __half pow(__half a, double b)
{
    return __float2half(pow(__half2float(a), b));
}
inline __device__ __half pow(__half a, __half b)
{
    return __float2half(pow(__half2float(a), __half2float(b)));
}
#endif
