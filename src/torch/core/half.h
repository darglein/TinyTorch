#pragma once

#include "torch/tiny_torch_config.h"

namespace tinytorch
{

struct Half
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