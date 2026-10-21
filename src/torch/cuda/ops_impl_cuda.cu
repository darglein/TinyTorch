
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
    // the memset writes +0.0, so -0.0 takes the kernel path to keep its sign
    if (value == 0 && !std::signbit(value) && a.is_contiguous())
    {
        cuda::DeviceGuard guard(a.device());
        TT_CHECK_CUDA_ERROR(
            cudaMemsetAsync(a.data_ptr(), 0, a.numel() * a.element_size(), cuda::getCurrentCUDAStream()));
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
    else if constexpr (std::is_same_v<TTarget, uint8_t> || std::is_same_v<TTarget, int16_t> ||
                       std::is_same_v<TTarget, uint16_t>)
    {
        // convert through int like the CPU, which wraps negative values (a direct conversion saturates on the GPU)
        b[i] = TTarget((int)a[i]);
    }
    else
    {
        b[i] = TTarget(a[i]);
    }
}


void copy_and_convert_impl(Tensor src, Tensor& target)
{
    CHECK_EQ(src.device(), target.device());
    // before the memcpy fast path, which would otherwise write past the end of a smaller target
    CHECK_EQ(src.numel(), target.numel());
    if (src.dtype() == target.dtype() && src.is_contiguous() && target.is_contiguous())
    {
        // trivial copy without conversion
        cuda::DeviceGuard guard(src.device());
        TT_CHECK_CUDA_ERROR(cudaMemcpyAsync(target.data_ptr(), src.data_ptr(), src.numel() * src.element_size(),
                                            cudaMemcpyDeviceToDevice, cuda::getCurrentCUDAStream()));
        return;
    }

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
        case kInt16:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.device(), src.scalar_type(), int16_t, src.numel(), copy_and_convert_impl,
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

__device__ inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// xorshift64* generator, one per thread. The state is a hash of all 64 seed bits and the thread index and never 0.
struct DeviceRng
{
    uint64_t state;

    __device__ DeviceRng(uint64_t seed, int64_t i)
    {
        state = splitmix64(seed ^ splitmix64((uint64_t)i));
        if (state == 0) state = 0x9E3779B97F4A7C15ULL;
    }

    __device__ uint32_t next()
    {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return (uint32_t)((state * 0x2545F4914F6CDD1DULL) >> 32);
    }

    __device__ uint64_t next64() { return ((uint64_t)next() << 32) | (uint64_t)next(); }

    // uniform in [0, 1)
    __device__ float uniform_float() { return float(next() >> 8) * (1.0f / 16777216.0f); }
    __device__ double uniform_double() { return double(next64() >> 11) * (1.0 / 9007199254740992.0); }
};

static uint64_t next_cuda_seed()
{
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    return dist(generator());
}

template <typename T>
__launch_bounds__(128) static __global__
    void rand_float_impl(TensorInfoCuda<T, -1> a, double low, double high, uint64_t seed)
{
    using G           = typename CpuComputeFloatType<T>::Type;
    using M           = typename MathComputeType<G>::Type;
    int64_t i         = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    int64_t grid_size = (int64_t)blockDim.x * gridDim.x;

    DeviceRng rng(seed, i);
    const M low_m  = M(low);
    const M high_m = M(high);

    for (; i < a.numel(); i += grid_size)
    {
        T v = T(low_m);
        if (low_m < high_m)
        {
            // redraw in the rare case that rounding produces high, so that the result is in [low, high)
            do
            {
                M x = std::is_same<M, double>::value ? M(rng.uniform_double()) : M(rng.uniform_float());
                v   = T(G(low_m + x * (high_m - low_m)));
            } while (!(M(G(v)) < high_m));
        }
        a[i] = v;
    }
}

void uniform_impl(Tensor& a, double mi, double ma)
{
    uint64_t seed       = next_cuda_seed();
    int64_t max_threads = std::min<int64_t>(a.numel(), int64_t(1024) * 1024 * 1024);
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), max_threads, rand_float_impl, a, mi, ma, seed);
}

template <typename T>
__launch_bounds__(128) static __global__
    void rand_int_impl(TensorInfoCuda<T> a, int64_t low, int64_t high, uint64_t seed)
{
    int64_t i         = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    int64_t grid_size = (int64_t)blockDim.x * gridDim.x;

    DeviceRng rng(seed, i);
    uint64_t range = (uint64_t)(high - low);

    for (; i < a.numel(); i += grid_size)
    {
        // multiply-shift range mapping
        uint64_t val = __umul64hi(rng.next64(), range);
        a[i]         = T((int64_t)((uint64_t)low + val));
    }
}

void uniform_int_impl(Tensor& a, int64_t low, int64_t high)
{
    CHECK_LT(low, high);
    uint64_t seed       = next_cuda_seed();
    int64_t max_threads = std::min<int64_t>(a.numel(), int64_t(1024) * 1024 * 1024);
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), max_threads, rand_int_impl, a, low, high, seed);
}


template <typename T>
__launch_bounds__(128) static __global__ void normal_random_float_impl(TensorInfoCuda<T, -1> a, uint64_t seed)
{
    int64_t i         = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    int64_t grid_size = (int64_t)blockDim.x * gridDim.x;

    DeviceRng rng(seed, i);
    const float two_pi = 6.28318530717958647692f;

    for (; i < a.numel(); i += grid_size)
    {
        // Box-Muller, u1 in (0, 1] so that log(u1) is finite
        float u1 = 1.0f - rng.uniform_float();
        float u2 = rng.uniform_float();
        float z0 = sqrtf(-2.0f * logf(u1)) * cosf(two_pi * u2);
        a[i]     = T(z0);
    }
}

void normal_random_impl(Tensor& a)
{
    uint64_t seed       = next_cuda_seed();
    int64_t max_threads = std::min<int64_t>(a.numel(), int64_t(1024) * 1024 * 1024);
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), max_threads, normal_random_float_impl, a, seed);
}

// mirrors cpu_impl::add_poisson_noise_impl (gaussian approximation for lambda > 15,
// Knuth's method below that)
template <typename T>
__launch_bounds__(128) static __global__ void poisson_noise_impl(TensorInfoCuda<T> a, uint64_t seed)
{
    int64_t i         = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    int64_t grid_size = (int64_t)blockDim.x * gridDim.x;

    DeviceRng rng(seed, i);

    for (; i < a.numel(); i += grid_size)
    {
        float lambda = float(a[i]);

        if (!(lambda > 0.0f))
        {
            a[i] = T(0.0);
            continue;
        }

        if (lambda > 15.0f)
        {
            // Box-Muller standard normal from two uniforms
            float u1 = 1.0f - rng.uniform_float();
            float u2 = rng.uniform_float();
            float n  = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979323846f * u2);

            float sample = lambda + std::sqrt(lambda) * n;
            a[i]         = T(std::max(0.0f, std::round(sample)));
        }
        else
        {
            // Knuth's method
            float L = std::exp(-lambda);
            float p = 1.0f;
            int k   = 0;

            do
            {
                k++;
                p *= rng.uniform_float();
            } while (p > L);

            a[i] = T(k - 1);
        }
    }
}
void add_poisson_noise_impl(Tensor& a)
{
    uint64_t seed       = next_cuda_seed();
    int64_t max_threads = std::min<int64_t>(a.numel(), int64_t(1024) * 1024 * 1024);
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), max_threads, poisson_noise_impl, a, seed);
}

template <typename T>
__launch_bounds__(128) static __global__ void clamp_impl_(TensorInfoCuda<T> src, double low, double high)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= src.numel()) return;

    // a non-finite bound means unbounded (std::numeric_limits is not specialized for __half)
    using G = typename CpuComputeFloatType<T>::Type;
    G v     = G(src[i]);
    if (::isfinite(low)) v = std::max(v, G(clamp_bound_to<T>(low, true)));
    if (::isfinite(high)) v = std::min(v, G(clamp_bound_to<T>(high, false)));
    src[i] = T(v);
}
void clamp_impl_(Tensor& a, double low, double high)
{
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), a.numel(), clamp_impl_, a, low, high);
}
template <typename T>
__launch_bounds__(128) static __global__
    void padding_2d_impl(TensorInfoCuda<T, 4> src, TensorInfoCuda<T, 4> dst, int pad_left, int pad_right, int pad_top,
                         int pad_bottom, PaddingMode mode)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= dst.numel()) return;

    int64_t x = i % dst.size(3);
    int64_t y = (i / dst.size(3)) % dst.size(2);
    int64_t c = i / dst.size(3) / dst.size(2) % dst.size(1);
    int64_t b = i / dst.size(3) / dst.size(2) / dst.size(1);

    auto x_src = x - pad_left;
    auto y_src = y - pad_top;

    if (mode == kZero)
    {
        if (x_src < 0 || x_src >= src.size(3) || y_src < 0 || y_src >= src.size(2))
        {
            dst(b, c, y, x) = T(0);
            return;  // Thread done for this padding element
        }
    }
    else if (mode == kReflect)
    {
        if (x_src < 0) x_src = -x_src;
        if (y_src < 0) y_src = -y_src;

        if (x_src >= src.size(3)) x_src -= 2 * (x_src - (src.size(3) - 1));
        if (y_src >= src.size(2)) y_src -= 2 * (y_src - (src.size(2) - 1));
    }

    // Clamp applied for kBorder (replicate), and as a safety fallback for extreme kReflect bounds
    if (x_src < 0) x_src = 0;
    if (x_src >= src.size(3)) x_src = src.size(3) - 1;

    if (y_src < 0) y_src = 0;
    if (y_src >= src.size(2)) y_src = src.size(2) - 1;

    dst(b, c, y, x) = src(b, c, y_src, x_src);
}

void padding_2d_impl(Tensor src, Tensor result, int pad_left, int pad_right, int pad_top, int pad_bottom,
                     PaddingMode mode)
{
    CUDA_SWITCH_MACRO_ALL(src.device(), src.scalar_type(), result.numel(), padding_2d_impl, src, result, pad_left,
                          pad_right, pad_top, pad_bottom, mode);
}

template <typename T>
__launch_bounds__(128) static __global__
    void padding_3d_impl(TensorInfoCuda<T, 5> src, TensorInfoCuda<T, 5> dst, int pad_left, int pad_right, int pad_top,
                         int pad_bottom, int pad_front, int pad_back, PaddingMode mode)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= dst.numel()) return;

    int64_t b = i / dst.size(4) / dst.size(3) / dst.size(2) / dst.size(1);
    int64_t c = i / dst.size(4) / dst.size(3) / dst.size(2) % dst.size(1);
    int64_t z = (i / dst.size(4) / dst.size(3)) % dst.size(2);
    int64_t y = (i / dst.size(4)) % dst.size(3);
    int64_t x = i % dst.size(4);

    // Offset to original source coordinates
    auto x_src = x - pad_left;
    auto y_src = y - pad_top;
    auto z_src = z - pad_front;

    if (mode == kZero)
    {
        if (x_src < 0 || x_src >= src.size(4) || y_src < 0 || y_src >= src.size(3) || z_src < 0 || z_src >= src.size(2))
        {
            dst(b, c, z, y, x) = T(0);
            return;  // Thread done for this padding element
        }
    }
    else if (mode == kReflect)
    {
        // Lower Boundary Reflection
        if (x_src < 0) x_src = -x_src;
        if (y_src < 0) y_src = -y_src;
        if (z_src < 0) z_src = -z_src;

        // Upper Boundary Reflection
        if (x_src >= src.size(4)) x_src -= 2 * (x_src - (src.size(4) - 1));
        if (y_src >= src.size(3)) y_src -= 2 * (y_src - (src.size(3) - 1));
        if (z_src >= src.size(2)) z_src -= 2 * (z_src - (src.size(2) - 1));
    }

    // Clamp applied for kBorder (replicate), and as a safety fallback for extreme kReflect bounds
    if (x_src < 0) x_src = 0;
    if (x_src >= src.size(4)) x_src = src.size(4) - 1;

    if (y_src < 0) y_src = 0;
    if (y_src >= src.size(3)) y_src = src.size(3) - 1;

    if (z_src < 0) z_src = 0;
    if (z_src >= src.size(2)) z_src = src.size(2) - 1;

    // Map the source voxel to the destination padding
    dst(b, c, z, y, x) = src(b, c, z_src, y_src, x_src);
}

void padding_3d_impl(Tensor src, Tensor result, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front,
                     int pad_back, PaddingMode mode)
{
    CUDA_SWITCH_MACRO_ALL(src.device(), src.scalar_type(), result.numel(), padding_3d_impl, src, result, pad_left,
                          pad_right, pad_top, pad_bottom, pad_front, pad_back, mode);
}

template <typename T>
__launch_bounds__(128) static __global__
    void repeat_interleave_impl(TensorInfoCuda<T> input, int64_t count, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    auto index_result    = result.LinearIndexToDimIndex(i);
    auto index_input     = index_result;
    index_input[0]       = index_result[0] / count;
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


template <typename T>
__launch_bounds__(128) static __global__
    void matmul_impl_kernel(TensorInfoCuda<T> a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    using G = std::conditional_t<std::is_same_v<T, __half>, float, typename CpuComputeFloatType<T>::Type>;

    int a_dim   = a.dim();
    int b_dim   = b.dim();
    int res_dim = result.dim();

    int64_t K = a.size(a_dim - 1);

    auto idx_res = result.LinearIndexToDimIndex(i);

    // batch dims are right-aligned and broadcast where the operand has size 1
    decltype(idx_res) idx_a, idx_b;
    idx_a.zero_();
    idx_b.zero_();
    for (int d = 0; d < a_dim - 2; ++d)
    {
        idx_a[d] = a.size(d) == 1 ? 0 : idx_res[res_dim - a_dim + d];
    }
    for (int d = 0; d < b_dim - 2; ++d)
    {
        idx_b[d] = b.size(d) == 1 ? 0 : idx_res[res_dim - b_dim + d];
    }
    idx_a[a_dim - 2] = idx_res[res_dim - 2];
    idx_b[b_dim - 1] = idx_res[res_dim - 1];

    G sum = G(0);
    for (int64_t k = 0; k < K; ++k)
    {
        idx_a[a_dim - 1] = k;
        idx_b[b_dim - 2] = k;

        sum += G(a[idx_a]) * G(b[idx_b]);
    }

    result[idx_res] = T(sum);
}

void matmul_impl(Tensor a, Tensor b, Tensor result)
{
    // Launch a 1D grid with a thread for every element in the output matrix
    CUDA_SWITCH_MACRO_ALL(a.device(), a.scalar_type(), result.numel(), matmul_impl_kernel, a, b, result);
}

}  // namespace cuda_impl
}  // namespace tinytorch