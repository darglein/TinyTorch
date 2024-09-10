
#include "torch/core/ops/ops_impl.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


namespace tinytorch
{
namespace cuda_impl
{

template <typename T>
__launch_bounds__(128) static __global__ void range_impl(TensorInfoCuda<T> a, double start, double end, double step)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(start + i * step);
}

void range_impl(Tensor a, double start, double end, double step)
{
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), a.numel(), range_impl, a, start, end, step);
}

template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfoCuda<T> a, double value)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(value);
}
template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> value)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    a[i] = T(value[0]);
}
template <typename T>
__launch_bounds__(128) static __global__ void fill_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> values, int64_t dim)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    auto index_a      = a.LinearIndexToDimIndex(i);
    auto index_values = index_a;
    index_values[dim] = 0;
    a[index_a]        = values[index_values];
}
void fill_impl(Tensor& a, double value)
{
    if (value == 0 && a.is_contiguous())
    {
        cuda::DeviceGuard guard(a.device());
        CHECK_CUDA_ERROR(cudaMemsetAsync(a.data_ptr(), 0, a.numel() * a.element_size(), cuda::getCurrentCUDAStream()));
        return;
    }

    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), a.numel(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor value)
{
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), a.numel(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor values, int64_t dim)
{
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), a.numel(), fill_impl, a, values, dim);
}

template <typename T>
__launch_bounds__(128) static __global__
    void permute_impl(TensorInfoCuda<T> src, TensorInfoCuda<T> result, DimIndexStruct<25, int> new_dims)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= src.numel()) return;

    auto index_src    = src.LinearIndexToDimIndex(i);
    auto index_result = index_src;

    for (int d = 0; d < src.dim(); ++d)
    {
        // index_result[new_dims[d]] = index_src[d];
        index_result[d] = index_src[new_dims[d]];
    }
    result[index_result] = src[index_src];
}

void permute_impl(Tensor& src, Tensor result, SizeType new_dims)
{
    CUDA_SYNC_CHECK_ERROR();
    CUDA_SWITCH_MACRO_ALL(src.device(), src.scalar_type(), src.numel(), permute_impl, src, result,
                          DimIndexStruct<25, int>(new_dims.vec()));
}

template <typename TSource, typename TTarget>
__launch_bounds__(128) static __global__
    void copy_and_convert_impl(TensorInfoCuda<TSource> a, TensorInfoCuda<TTarget> b)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    if constexpr (std::is_same_v<TSource, int64_t>)
    {
        b[i] = TTarget((long long)a[i]);
    }
    else if constexpr (std::is_same_v<TTarget, int64_t>)
    {
        b[i] = TTarget((long long)a[i]);
    }
    else
    {
        b[i] = TTarget(a[i]);
    }
}


void copy_and_convert_impl(Tensor src, Tensor& target)
{
    CHECK_EQ(src.device(), target.device());
    if (src.dtype() == target.dtype() && src.is_contiguous() && target.is_contiguous())
    {
        // trivial copy without conversion
        cuda::DeviceGuard guard(src.device());
        CHECK_CUDA_ERROR(cudaMemcpyAsync(target.data_ptr(), src.data_ptr(), src.numel() * src.element_size(),
                                         cudaMemcpyDeviceToDevice, cuda::getCurrentCUDAStream()));
        return;
    }

    CHECK_EQ(src.numel(), target.numel());
    switch (target.dtype())
    {
        case kUInt8:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), uint8_t, src.numel(), copy_and_convert_impl,
                                       src, target);
            break;
        }
        case kUInt16:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), uint16_t, src.numel(), copy_and_convert_impl,
                                       src, target);
            break;
        }
        case kInt32:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), int32_t, src.numel(), copy_and_convert_impl,
                                       src, target);
            break;
        }
        case kInt64:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), int64_t, src.numel(), copy_and_convert_impl,
                                       src, target);
            break;
        }
        case kFloat16:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), half, src.numel(), copy_and_convert_impl, src,
                                       target);
            break;
        }
        case kFloat32:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), float, src.numel(), copy_and_convert_impl, src,
                                       target);
            break;
        }
        case kFloat64:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), double, src.numel(), copy_and_convert_impl, src,
                                       target);
            break;
        }
        default:
            throw std::runtime_error("invalid type");
    }
}

inline TT_HD unsigned int xorshift64(unsigned int x)
{
    //    x ^= x << 13;
    //    x ^= x >> 7;
    //    x ^= x << 17;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

template <typename T>
__launch_bounds__(128) static __global__
    void rand_float_impl(TensorInfoCuda<T, -1> a, float low, float high, uint64_t seed)
{
    int64_t i         = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    int64_t grid_size = blockDim.x * gridDim.x;
    //    if (i >= a.numel()) return;
    unsigned int seed2 = seed + (i * 146394585141533LL);

    for (; i < a.numel(); i += grid_size)
    {
        seed2    = xorshift64(seed2);
        float xf = float(seed2) * (1.0f / std::numeric_limits<unsigned int>::max());
        a[i]     = T(xf * (high - low) + low);
        //        a[i] = 0;
    }
}

void uniform_impl(Tensor& a, double mi, double ma)
{
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    uint64_t seed = dist(generator());

    int64_t max_threads = std::min<int64_t>(a.numel(), int64_t(1024) * 1024 * 1024);

    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), max_threads, rand_float_impl, a, (float)mi, (float)ma, seed);
}

template <typename T>
__launch_bounds__(128) static __global__ void rand_int_impl(TensorInfoCuda<T> a, int low, int high, uint64_t seed)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    auto seed2 = seed + i;
    uint64_t x = xorshift64(seed2);
    a[i]       = T((unsigned long long)(x % (high - low) + low));
}

void uniform_int_impl(Tensor& a, int low, int high)
{
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    uint64_t seed = dist(generator());
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), a.numel(), rand_int_impl, a, low, high, seed);
}

template <typename T>
__launch_bounds__(128) static __global__ void clamp_impl_(TensorInfoCuda<T> src, double low, double high)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= src.numel()) return;

    using G  = typename CpuComputeFloatType<T>::Type;
    T low_t  = ::isfinite(low) ? T(low) : std::numeric_limits<T>::lowest();
    T high_t = ::isfinite(high) ? T(high) : std::numeric_limits<T>::max();

    {
        src[i] = std::min(G(high_t), std::max(G(src[i]), G(low_t)));
    }
}
void clamp_impl_(Tensor& a, double low, double high)
{
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), a.numel(), clamp_impl_, a, low, high);
}

template <typename T>
__launch_bounds__(128) static __global__
    void repeat_interleave_impl(TensorInfoCuda<T> input, int64_t count, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    auto index_result    = result.LinearIndexToDimIndex(i);
    auto index_input     = input.LinearIndexToDimIndex(i / count);
    result[index_result] = input[index_input];
}
void repeat_interleave_impl(Tensor input, int64_t count, Tensor result)
{
    CUDA_SWITCH_MACRO_ALL(result.device(), result.scalar_type(), result.numel(), repeat_interleave_impl, input, count,
                          result);
}


template <typename T>
__launch_bounds__(128) static __global__ void repeat_impl(TensorInfoCuda<T> src, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    {
        auto index_result = result.LinearIndexToDimIndex(i);
        auto index_src    = index_result;
        for (int d = 0; d < src.dim(); ++d)
        {
            index_src[d] = index_result[d] % src.size(d);
        }
        result[i] = src[index_src];
    }
}
void repeat_impl(Tensor t, SizeType sizes, Tensor result)
{
    CUDA_SWITCH_MACRO_ALL(t.device(), t.scalar_type(), result.numel(), repeat_impl, t, result);
}


}  // namespace cuda_impl
}  // namespace tinytorch