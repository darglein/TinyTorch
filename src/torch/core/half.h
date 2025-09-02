#pragma once

#include "torch/tiny_torch_config.h"

namespace tinytorch
{

struct TINYTORCH_API Half
{
    uint16_t h;

    Half() {}
    Half(float f);
    Half(double d) : Half(float(d)) {}
    Half(uint16_t i);

    operator float();
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


namespace tinytorch
{


inline uint32_t half_to_float(uint16_t h2)
{
    union FP32
    {
        uint32_t u;
        float f;
        struct
        {
            uint32_t Mantissa : 23;
            uint32_t Exponent : 8;
            uint32_t Sign : 1;
        };
    };

    union FP16
    {
        uint16_t u;
        struct
        {
            uint16_t Mantissa : 10;
            uint16_t Exponent : 5;
            uint16_t Sign : 1;
        };
    };


    FP16 h;
    h.u                          = h2;
    static const FP32 magic      = {(254 - 15) << 23};
    static const FP32 was_infnan = {(127 + 16) << 23};
    FP32 o;

    o.u = (h.u & 0x7fff) << 13;  // exponent/mantissa bits
    o.f *= magic.f;              // exponent adjust
    if (o.f >= was_infnan.f)     // make sure Inf/NaN survive
        o.u |= 255 << 23;
    o.u |= (h.u & 0x8000) << 16;  // sign bit
    return o.u;
}


inline Half::operator float()
{
    uint32_t f = half_to_float(h);
    return *(float*)&f;
}
}  // namespace tinytorch



namespace tinytorch
{

// https://gist.github.com/neshume/0edc6ae1c5ad332bb4c62026be68a2fb
// float16
// Martin Kallman
//
// Fast single-precision to half-precision floating point conversion
//  - Supports signed zero, denormals-as-zero (DAZ), flush-to-zero (FTZ),
//    clamp-to-max
//  - Does not support infinities or NaN
//  - Few, partially pipelinable, non-branching instructions,
//  - Core opreations ~10 clock cycles on modern x86-64
inline uint16_t half_from_float(uint32_t inu)
{
    //    uint32_t inu = *((uint32_t*)&in);
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = inu & 0x7fffffff;  // Non-sign bits
    t2 = inu & 0x80000000;  // Sign bit
    t3 = inu & 0x7f800000;  // Exponent

    t1 >>= 13;  // Align mantissa on MSB
    t2 >>= 16;  // Shift sign bit into position

    t1 -= 0x1c000;  // Adjust bias

    t1 = (t3 < 0x38800000) ? 0 : t1;
    t1 = (t1 > 0x7bff) ? 0x7bff : t1;
    t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero

    t1 |= t2;  // Re-insert sign bit

    return t1;
};

inline Half::Half(float f)
{
    h = half_from_float(*(uint32_t*)&f);
}

inline Half::Half(uint16_t i)
{
    h = i;
}
}  // namespace tinytorch
