
#include "torch/core/ops/ops_impl.h"
#include "torch/cuda/atomic_minmax.h"
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
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), range_impl, a, start, end, step);
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
__launch_bounds__(128) static __global__ void fill_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> values, int dim)
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
        cudaMemset(a.data_ptr(), 0, a.numel() * a.element_size());
        return;
    }

    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor value)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor values, int dim)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), fill_impl, a, values, dim);
}

template <typename T>
__launch_bounds__(128) static __global__
    void permute_impl(TensorInfoCuda<T> src, TensorInfoCuda<T> result, DimIndexStruct<25> new_dims)
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

void permute_impl(Tensor& src, Tensor& result, SizeType new_dims)
{
    CUDA_SYNC_CHECK_ERROR();
    CUDA_SWITCH_MACRO_ALL(src.scalar_type(), src.numel(), permute_impl, src, result, new_dims.vec());
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
    CHECK_EQ(src.numel(), target.numel());
    switch (target.dtype())
    {
        case kInt32:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int32_t, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        case kInt64:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int64_t, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        case kFloat16:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), half, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        case kFloat32:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), float, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        case kFloat64:
        {
            CUDA_SWITCH_MACRO_ALL_DUAL(src.scalar_type(), double, src.numel(), copy_and_convert_impl, src, target);
            break;
        }
        default:
            throw std::runtime_error("invalid type");
    }
}

inline TT_HD uint64_t xorshift64(uint64_t x)
{
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

template <typename T>
__launch_bounds__(128) static __global__ void rand_float_impl(TensorInfoCuda<T> a, float low, float high, uint64_t seed)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    auto seed2 = seed + i;
    uint64_t x = xorshift64(seed2);
    double xf  = double(x) / std::numeric_limits<uint64_t>::max();

    a[i] = T(xf * (high - low) + low);
}

void uniform_impl(Tensor& a, double mi, double ma)
{
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    uint64_t seed = dist(generator());
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), rand_float_impl, a, (float)mi, (float)ma, seed);
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
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), rand_int_impl, a, low, high, seed);
}


template <typename T>
__launch_bounds__(128) static __global__ void sum_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    atomicAdd(&result[0], a[i]);
}

void sum_impl(Tensor a, Tensor& result)
{
    auto stream = cuda::getCurrentCUDAStream();
    switch (a.scalar_type())
    {
        CUDA_CASE_MACRO(sum_impl<int32_t>, kInt32, a.numel(), a, result)
        CUDA_CASE_MACRO(sum_impl<half>, kFloat16, a.numel(), a, result)
        CUDA_CASE_MACRO(sum_impl<float>, kFloat, a.numel(), a, result)
        CUDA_CASE_MACRO(sum_impl<double>, kDouble, a.numel(), a, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}

template <typename T>
__launch_bounds__(128) static __global__ void sum_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
{
    int64_t linear_index_input = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (linear_index_input >= input.numel()) return;

    auto index_input  = input.LinearIndexToDimIndex(linear_index_input);
    auto result_index = index_input;
    result_index[dim] = 0;
    atomicAdd(&result[result_index], input[index_input]);
}

void sum_impl(Tensor a, int64_t dim, Tensor& result)
{
    auto stream = cuda::getCurrentCUDAStream();
    switch (a.scalar_type())
    {
        CUDA_CASE_MACRO(sum_impl<int32_t>, kInt32, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<half>, kHalf, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<float>, kFloat, a.numel(), a, dim, result)
        CUDA_CASE_MACRO(sum_impl<double>, kDouble, a.numel(), a, dim, result)
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }
}



template <typename T>
__launch_bounds__(128) static __global__ void prod_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    {
        auto index_result = result.LinearIndexToDimIndex(i);

        T prod = T(1.f);
        for (int64_t j = 0; j < input.size(dim); ++j)
        {
            auto index_input = index_result;
            index_input[dim] = j;
            prod             = prod * input[index_input];
        }
        result[index_result] = prod;
    }
}

void prod_impl(Tensor input, int64_t dim, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(input.scalar_type(), result.numel(), prod_impl, input, dim, result);
}
template <typename T>
__launch_bounds__(128) static __global__
    void cumprod_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    {
        auto index_result = result.LinearIndexToDimIndex(i);

        T prod = T(1.f);
        for (int64_t j = 0; j <= index_result[dim]; ++j)
        {
            auto index_input = index_result;
            index_input[dim] = j;
            prod             = prod * input[index_input];
        }
        result[index_result] = prod;
    }
}
void cumprod_impl(Tensor input, int64_t dim, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(input.scalar_type(), result.numel(), cumprod_impl, input, dim, result);
}
template <typename T>
__launch_bounds__(128) static __global__
    void cumsum_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;
    {
        auto index_result = result.LinearIndexToDimIndex(i);

        T prod = T(0.f);
        for (int64_t j = 0; j <= index_result[dim]; ++j)
        {
            auto index_input = index_result;
            index_input[dim] = j;
            prod             = prod + input[index_input];
        }
        result[index_result] = prod;
    }
}
void cumsum_impl(Tensor input, int64_t dim, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(input.scalar_type(), result.numel(), cumsum_impl, input, dim, result);
}

template <typename T>
__launch_bounds__(128) static __global__
    void min_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MIN(a[i], b[i]);
}

template <typename T>
__launch_bounds__(128) static __global__ void min_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;
    atomicMin(&result[0], a[i]);
}


void min_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), min_impl, a, b, result);
}
void min_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), min_impl, a, result);
}


template <typename T>
__launch_bounds__(128) static __global__ void max_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;
    atomicMax(&result[0], a[i]);
}


template <typename T>
__launch_bounds__(128) static __global__
    void max_impl(TensorInfoCuda<T> a, TensorInfoCuda<T> b, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= a.numel()) return;

    result[i] = MAX(a[i], b[i]);
}

void max_impl(Tensor a, Tensor b, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), max_impl, a, b, result);
}
void max_impl(Tensor a, Tensor& result)
{
    CUDA_SWITCH_MACRO_FLOAT(a.scalar_type(), a.numel(), max_impl, a, result);
}


template <typename T>
__launch_bounds__(128) static __global__
    void min_max_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<int64_t> indices, TensorInfoCuda<T> result,
                      bool calc_min)
{
    using G = typename CpuComputeFloatType<T>::Type;


    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= input.numel()) return;

    {
        G v               = input[i];
        auto index_input  = input.LinearIndexToDimIndex(i);
        auto index_result = index_input;
        index_result[dim] = 0;

        auto& result_value = result[index_result];
        auto& result_index = indices[index_result];


        if (calc_min)
        {
            atomicMin(&result_value, v);
        }
        else
        {
            atomicMax(&result_value, v);
        }
    }
}
void min_impl(Tensor input, int64_t dim, Tensor& result, Tensor& indices)
{
    CUDA_SWITCH_MACRO_FLOAT(input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, true);
    indices = Tensor();
}

void max_impl(Tensor input, int64_t dim, Tensor& result, Tensor& indices)
{
    CUDA_SWITCH_MACRO_FLOAT(input.scalar_type(), input.numel(), min_max_impl, input, dim, indices, result, false);
    indices = Tensor();
}

template <typename T, typename TIndex>
__launch_bounds__(128) static __global__
    void index_select_impl(TensorInfoCuda<T> input, int64_t dim, TensorInfoCuda<TIndex, 1> index,
                           TensorInfoCuda<T> result)
{
    int64_t result_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (result_linear_index >= result.numel()) return;

    auto index_result    = result.LinearIndexToDimIndex(result_linear_index);
    auto index_input     = index_result;
    index_input[dim]     = index[index_result[dim]];
    result[index_result] = input[index_input];
}

template <typename TIndex>
static void index_select_helper(Tensor input, int64_t dim, TensorInfoCuda<TIndex, 1> index, Tensor result)
{
    CUDA_SWITCH_MACRO_ALL(result.scalar_type(), result.numel(), index_select_impl, input, dim, index, result);
}

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor& result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_select_helper<int32_t>(input, dim, index, result);
            break;
        case kLong:
            index_select_helper<int64_t>(input, dim, index, result);
            break;
    }
}



template <typename T, typename TIndex>
__launch_bounds__(128) static __global__
    void index_add_impl(int64_t dim, TensorInfoCuda<TIndex> index, TensorInfoCuda<T> data, TensorInfoCuda<T> result)
{
    int64_t input_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (input_linear_index >= data.numel()) return;

    auto index_input  = data.LinearIndexToDimIndex(input_linear_index);
    auto index_result = index_input;
    CUDA_KERNEL_ASSERT(index_input[dim] < index.sizes[0]);
    index_result[dim] = index[index_input[dim]];
    CUDA_KERNEL_ASSERT(index_result[dim] < result.sizes[dim]);

    CUDA_KERNEL_ASSERT(result.index_in_range(index_result));
    CUDA_KERNEL_ASSERT(data.index_in_range(index_input));
    atomicAdd(&result[index_result], data[index_input]);
}

template <typename TIndex>
static void index_add_helper(Tensor data, int64_t dim, TensorInfoCuda<TIndex> index, Tensor result)
{
    CUDA_SWITCH_MACRO_FLOAT(result.scalar_type(), data.numel(), index_add_impl, dim, index, data, result);
}

void index_add_impl(int64_t dim, Tensor index, Tensor data, Tensor& result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_add_helper<int32_t>(data, dim, index, result);
            break;
        case kLong:
            index_add_helper<int64_t>(data, dim, index, result);
            break;
    }
}


template <typename T>
__launch_bounds__(128) static __global__
    void gather_impl(TensorInfoCuda<T> data, int64_t dim, TensorInfoCuda<int64_t> index, TensorInfoCuda<T> result)
{
    int64_t input_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (input_linear_index >= result.numel()) return;

    {
        auto index_result = result.LinearIndexToDimIndex(input_linear_index);
        auto index_input  = index_result;

        index_input[dim] = index[index_result];

        result[index_result] = data[index_input];
    }
}
void gather_impl(Tensor data, int64_t dim, Tensor index, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(data.scalar_type(), result.numel(), gather_impl, data, dim, index, result);
}


template <typename T, typename TIndex>
__launch_bounds__(128) static __global__
    void index_copy_impl(TensorInfoCuda<T> target, int64_t dim, TensorInfoCuda<TIndex> index, TensorInfoCuda<T> value)
{
    int64_t input_linear_index = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (input_linear_index >= value.numel()) return;

    auto index_input  = value.LinearIndexToDimIndex(input_linear_index);
    auto index_result = index_input;
    CUDA_KERNEL_ASSERT(index_input[dim] < index.sizes[0]);
    index_result[dim] = index[index_input[dim]];
    CUDA_KERNEL_ASSERT(index_result[dim] < target.sizes[dim]);

    CUDA_KERNEL_ASSERT(target.index_in_range(index_result));
    CUDA_KERNEL_ASSERT(value.index_in_range(index_input));
    target[index_result] = value[index_input];
}

template <typename TIndex>
static void index_copy_helper(Tensor& target, int64_t dim, TensorInfoCuda<TIndex> index, Tensor value)
{
    CUDA_SWITCH_MACRO_ALL(target.scalar_type(), value.numel(), index_copy_impl, target, dim, index, value);
}

void index_copy_impl(Tensor& target, int64_t dim, Tensor index, Tensor value)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_copy_helper<int32_t>(target, dim, index, value);
            break;
        case kLong:
            index_copy_helper<int64_t>(target, dim, index, value);
            break;
        default:
            throw std::runtime_error("invalid type");
    }
}



template <typename T>
__launch_bounds__(128) static __global__
    void transpose_impl(TensorInfoCuda<T> input, int64_t dim0, int64_t dim1, TensorInfoCuda<T> result)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    auto index_result = result.LinearIndexToDimIndex(i);
    auto index_input  = index_result;
    // swap(index_input[dim0], index_input[dim1]);
    auto t            = index_input[dim0];
    index_input[dim0] = index_input[dim1];
    index_input[dim1] = t;


    result[index_result] = input[index_input];
}
void transpose_impl(Tensor input, int64_t dim0, int64_t dim1, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(result.scalar_type(), result.numel(), transpose_impl, input, dim0, dim1, result);
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
    CUDA_SWITCH_MACRO_ALL(a.scalar_type(), a.numel(), clamp_impl_, a, low, high);
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
void repeat_interleave_impl(Tensor input, int64_t count, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(result.scalar_type(), result.numel(), repeat_interleave_impl, input, count, result);
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
void repeat_impl(Tensor t, SizeType sizes, Tensor& result)
{
    CUDA_SWITCH_MACRO_ALL(t.scalar_type(), result.numel(), repeat_impl, t, result);
}


}  // namespace cuda_impl
}  // namespace tinytorch