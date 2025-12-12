/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl_cpu.h"

#include "ops_impl_cpu_helper.h"
#include "torch/core/ops/ops_impl.h"
#include "torch/core/tensor_info.h"
#include "torch/cuda/cached_memory_allocator.h"
#include "torch/cuda/ops_impl_cuda_helper.h"



namespace tinytorch
{
namespace cpu_impl
{

template <typename T>
static void print_impl(std::ostream& strm, TensorInfo<T> a)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        strm << std::setw(std::cout.precision() + 6) << a[i] << " ";
    }
}

void print_impl(std::ostream& strm, Tensor t)
{
    print_impl<double>(strm, t.to(kDouble));
}

void to_impl_cpu_cuda(Tensor src, Tensor dest, bool async)
{
#ifdef TT_HAS_CUDA
    CHECK(src.is_contiguous()) << src.sizes() << " " << src.strides();
    CHECK(dest.is_contiguous()) << dest.sizes() << " " << dest.strides();
    CHECK_EQ(src.numel(), dest.numel());
    CHECK_EQ(src.scalar_type(), dest.scalar_type());

    if (async)
    {
        //        if (src.is_cpu())
        //        {
        //            CHECK(src.options().pinned_memory_);
        //        }
        //        if (dest.is_cpu())
        //        {
        //            CHECK(dest.options().pinned_memory_);
        //        }
        if (src.is_cpu() && !src.options().pinned_memory_)
        {
            async = false;
        }
        if (dest.is_cpu() && !dest.options().pinned_memory_)
        {
            async = false;
        }
    }

    int64_t bytes = src.element_size() * src.numel();

    if (src.is_cpu() || dest.is_cpu())
    {
        auto type = (dest.device() == kCPU) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        cuda::DeviceGuard guard(src.device() == kCPU ? dest.device() : src.device());

        if (async)
        {
            auto strm = cuda::getCurrentCUDAStream();
            TT_CHECK_CUDA_ERROR(cudaMemcpyAsync(dest.data_ptr(), src.data_ptr(), bytes, type, strm));
        }
        else
        {
            TT_CHECK_CUDA_ERROR(cudaMemcpy(dest.data_ptr(), src.data_ptr(), bytes, type));
        }
    }
    else
    {
        // p2p copy
        bool use_uva = dest.is_uva() && src.is_uva();
        if (async)
        {
            cudaEvent_t src_buffer_ready, mempcy_finished;
            {
                cuda::DeviceGuard guard(src.device());
                src_buffer_ready = cuda::getNextEvent();
                TT_CHECK_CUDA_ERROR(cudaEventRecord(src_buffer_ready, cuda::getCurrentCUDAStream()));
            }


            {
                cuda::DeviceGuard guard(dest.device());
                TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(cuda::getCurrentCUDAStream(), src_buffer_ready));

                if (use_uva)
                {
                    TT_CHECK_CUDA_ERROR(cudaMemcpyAsync(dest.data_ptr(), src.data_ptr(), bytes, cudaMemcpyDefault,
                                                        cuda::getCurrentCUDAStream()));
                }
                else
                {
                    TT_CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(dest.data_ptr(), dest.device().index(), src.data_ptr(),
                                                            src.device().index(), bytes, cuda::getCurrentCUDAStream()));
                }
                mempcy_finished = cuda::getNextEvent();
                TT_CHECK_CUDA_ERROR(cudaEventRecord(mempcy_finished, cuda::getCurrentCUDAStream()));
            }

            {
                cuda::DeviceGuard guard(src.device());
                TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(cuda::getCurrentCUDAStream(), mempcy_finished));
            }
        }
        else
        {
            cuda::DeviceGuard guard(dest.device());
            if (use_uva)
            {
                TT_CHECK_CUDA_ERROR(cudaMemcpy(dest.data_ptr(), src.data_ptr(), bytes, cudaMemcpyDefault));
            }
            else
            {
                TT_CHECK_CUDA_ERROR(cudaMemcpyPeer(dest.data_ptr(), dest.device().index(), src.data_ptr(),
                                                   src.device().index(), bytes));
            }
        }
    }


#else
    CHECK(false);
#endif
}

template <typename T>
static void sort_impl(TensorInfo<T> a, int64_t dim, TensorInfo<T> out_t, TensorInfo<int64_t> out_index)
{
    using G = typename CpuComputeFloatType<T>::Type;
#if 0
    std::vector<std::pair<G, int64_t>> data;
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        data.push_back({a[i], i});
    }
    std::sort(data.begin(), data.end());

    for (int64_t i = 0; i < a.numel(); ++i)
    {
        out_t[i]     = data[i].first;
        out_index[i] = data[i].second;
    }
#else
    int64_t size_along_dim = out_t.size(dim);

    std::vector<std::pair<G, int64_t>> data;
    data.resize(size_along_dim);

    for (int64_t i = 0; i < out_t.numel() / size_along_dim; ++i)
    {
        auto index = a.LinearIndexToDimIndexSkipDim(i, dim);
        for (int64_t o = 0; o < size_along_dim; ++o)
        {
            index[dim] = o;
            data[o]    = {a[index], o};
        }
        std::sort(data.begin(), data.end());
        for (int64_t o = 0; o < size_along_dim; ++o)
        {
            index[dim]       = o;
            out_t[index]     = data[o].first;
            out_index[index] = data[o].second;
        }
    }
#endif
}
void sort_impl(Tensor a, int64_t dim, Tensor& out_t, Tensor& out_index)
{
    // CHECK_EQ(a.dim(), 1);
    SWITCH_MACRO_ALL(a.scalar_type(), sort_impl, a, dim, out_t, out_index);
}


template <typename T>
static void fill_impl(TensorInfo<T> a, double value)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = T(value);
    }
}
template <typename T>
static void fill_impl(TensorInfo<T> a, TensorInfo<T> value)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = value[0];
    }
}
template <typename T>
static void fill_impl(TensorInfo<T> a, TensorInfo<T> values, int64_t dim)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        auto index_a      = a.LinearIndexToDimIndex(i);
        auto index_values = index_a;
        index_values[dim] = 0;
        a[index_a]        = values[index_values];
    }
}

void fill_impl(Tensor& a, double value)
{
    if (value == 0 && a.is_contiguous())
    {
        memset(a.data_ptr(), 0, a.numel() * a.element_size());
        return;
    }
    SWITCH_MACRO_ALL(a.scalar_type(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor value)
{
    SWITCH_MACRO_ALL(a.scalar_type(), fill_impl, a, value);
}
void fill_impl(Tensor& a, Tensor values, int64_t dim)
{
    SWITCH_MACRO_ALL(a.scalar_type(), fill_impl, a, values, dim);
}

template <typename T>
static void permute_impl(TensorInfo<T> src, TensorInfo<T> result, SizeType new_dims)
{
    for (int64_t i = 0; i < src.numel(); ++i)
    {
        auto index_src    = src.LinearIndexToDimIndex(i);
        auto index_result = index_src;

        for (int d = 0; d < src.dim(); ++d)
        {
            // index_result[new_dims[d]] = index_src[d];
            index_result[d] = index_src[new_dims[d]];
        }
        result[index_result] = src[index_src];
    }
}


void permute_impl(Tensor& src, Tensor result, SizeType new_dims)
{
    SWITCH_MACRO_ALL(src.scalar_type(), permute_impl, src, result, new_dims);
}

template <typename T>
static void range_impl(TensorInfo<T> a, double start, double end, double step)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        a[i] = T(start + i * step);
    }
}
void range_impl(Tensor a, double start, double end, double step)
{
    SWITCH_MACRO_ALL(a.scalar_type(), range_impl, a, start, end, step);
}


template <typename T>
static void rand_float_impl(TensorInfo<T> t, std::mt19937& mersenne_engine, float low, float high)
{
    float scale = (1.f / float(std::numeric_limits<uint32_t>::max())) * (high - low);
    std::uniform_real_distribution<float> dist{low, high};

    auto N = t.numel();
    if (t.contiguous)
    {
        T* pt = t.data;
        for (int64_t i = 0; i < N; ++i)
        {
            uint32_t value = (uint32_t)mersenne_engine.operator()();
            float xf       = float(value) * scale;
            pt[i]          = T(xf + low);
        }
    }
    else
    {
        for (int64_t i = 0; i < N; ++i)
        {
            uint32_t value = (uint32_t)mersenne_engine.operator()();
            float xf       = float(value) * scale;
            t[i]           = T(xf + low);
        }
    }
}

void uniform_impl(Tensor& t, double mi, double ma)
{
    SWITCH_MACRO_ALL(t.scalar_type(), rand_float_impl, t, generator(), (float)mi, (float)ma);
}

template <typename T>
static void rand_int_impl(TensorInfo<T> t, std::mt19937& mersenne_engine, int64_t low, int64_t high)
{
    // the high bounds is exclusive
    std::uniform_int_distribution<int64_t> dist{low, high - 1};
    for (int64_t i = 0; i < t.numel(); ++i)
    {
        t[i] = T(dist(mersenne_engine));
    }
}

void uniform_int_impl(Tensor& t, int64_t low, int64_t high)
{
    SWITCH_MACRO_INT(t.scalar_type(), rand_int_impl, t, generator(), low, high);
}



template <typename T>
static void sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        result[0] = result[0] + a[i];
    }
}

void sum_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sum_impl, a, result);
}

template <typename T>
static void abs_sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    for (int64_t i = 0; i < a.numel(); ++i)
    {
        T v       = a[i];
        result[0] = result[0] + (v > T(0.) ? T(v) : T(-v));
    }
}

void abs_sum_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), abs_sum_impl, a, result);
}


template <typename T>
static void sum_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    for (int64_t linear_index_input = 0; linear_index_input < input.numel(); ++linear_index_input)
    {
        auto index_input     = input.LinearIndexToDimIndex(linear_index_input);
        auto result_index    = index_input;
        result_index[dim]    = 0;
        result[result_index] = result[result_index] + input[index_input];
    }
}

void sum_impl(Tensor a, int64_t dim, Tensor result)
{
    SWITCH_MACRO_FLOAT(a.scalar_type(), sum_impl, a, dim, result);
}


template <typename T>
static void prod_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    for (int64_t linear_index_input = 0; linear_index_input < input.numel(); ++linear_index_input)
    {
        auto index_input     = input.LinearIndexToDimIndex(linear_index_input);
        auto index_result    = index_input;
        index_result[dim]    = 0;
        result[index_result] = result[index_result] * input[index_input];
    }
}

void prod_impl(Tensor input, int64_t dim, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), prod_impl, input, dim, result);
}
template <typename T>
static void cumprod_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
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
void cumprod_impl(Tensor input, int64_t dim, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), cumprod_impl, input, dim, result);
}
template <typename T>
static void cumsum_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
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
void cumsum_impl(Tensor input, int64_t dim, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), cumsum_impl, input, dim, result);
}

template <typename T>
static void min_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;

    if (a.numel() < 128)
    {
        for (int64_t i = 0; i < a.numel(); ++i)
        {
            result[0] = std::min(G(a[i]), G(result[0]));
        }
    }
    else
    {
        G min_val = std::numeric_limits<G>::max();

        // #pragma omp parallel for reduction(min : min_val)
        for (int64_t i = 0; i < a.numel(); ++i)
        {
            G v = G(a[i]);
            if (v < min_val)
            {
                min_val = v;
            }
        }

        result[0] = min_val;
    }
}

void min_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl, a, result);
}


template <typename T>
static void max_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;
    if (a.numel() < 128)
    {
        for (int64_t i = 0; i < a.numel(); ++i)
        {
            result[0] = std::max(G(a[i]), G(result[0]));
        }
    }
    else
    {
        G max_val = std::numeric_limits<G>::lowest();

        // #pragma omp parallel for reduction(max : max_val)
        for (int64_t i = 0; i < a.numel(); ++i)
        {
            G v = G(a[i]);
            if (v > max_val)
            {
                max_val = v;
            }
        }

        result[0] = max_val;
    }
}

void max_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl, a, result);
}

template <typename T>
static void min_max_impl(TensorInfo<T> input, int64_t dim, TensorInfo<int64_t> indices, TensorInfo<T> result,
                         bool calc_min)
{
    using G = typename CpuComputeFloatType<T>::Type;

    auto op_min = std::less<G>();
    auto op_max = std::greater<G>();

    for (int64_t i = 0; i < input.numel(); ++i)
    {
        G v               = input[i];
        auto index_input  = input.LinearIndexToDimIndex(i);
        auto index_result = index_input;
        index_result[dim] = 0;

        auto& result_value = result[index_result];
        auto& result_index = indices[index_result];

        bool comp = calc_min ? op_min(v, result_value) : op_max(v, result_value);
        if (comp)
        {
            result_value = v;
            result_index = index_input[dim];
        }
    }
}
void min_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    SWITCH_MACRO_ALL(input.scalar_type(), min_max_impl, input, dim, indices, result, true);
}

void max_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    SWITCH_MACRO_ALL(input.scalar_type(), min_max_impl, input, dim, indices, result, false);
}


template <typename T, typename Indextype>
static void index_select_impl(TensorInfo<T> input, int64_t dim, TensorInfo<Indextype> index, TensorInfo<T> result)
{
    for (int64_t result_linear_index = 0; result_linear_index < result.numel(); ++result_linear_index)
    {
        auto index_result    = result.LinearIndexToDimIndex(result_linear_index);
        auto index_input     = index_result;
        index_input[dim]     = index[index_result[dim]];
        result[index_result] = input[index_input];
    }
}

void index_select_impl(Tensor input, int64_t dim, Tensor index, Tensor result)
{
    CHECK(input.is_cpu());
    CHECK_LT(dim, input.dim());


    switch (index.dtype())
    {
        case kInt32:
        {
            SWITCH_MACRO_ALL_DUAL(input.scalar_type(), int32_t, index_select_impl, input, dim, index, result);
            break;
        }
        case kInt64:
        {
            SWITCH_MACRO_ALL_DUAL(input.scalar_type(), int64_t, index_select_impl, input, dim, index, result);
            break;
        }
        default:
            throw std::runtime_error("invalid index type");
    }
}

template <typename T, typename TIndex>
static void index_add_impl(int64_t dim, TensorInfo<TIndex> index, TensorInfo<T> data, TensorInfo<T> result)
{
    for (int64_t input_linear_index = 0; input_linear_index < data.numel(); ++input_linear_index)
    {
        auto index_input = data.LinearIndexToDimIndex(input_linear_index);

        auto index_result = index_input;
        index_result[dim] = index[index_input[dim]];

        result[index_result] = result[index_result] + data[index_input];
    }
}

template <typename TIndex>
static void index_add_helper(int64_t dim, TensorInfo<TIndex> index, Tensor data, Tensor result)
{
    SWITCH_MACRO_ALL(data.scalar_type(), index_add_impl, dim, index, data, result);
}

void index_add_impl(int64_t dim, Tensor index, Tensor data, Tensor result)
{
    switch (index.scalar_type())
    {
        case kInt32:
            index_add_helper<int32_t>(dim, index, data, result);
            break;
        case kLong:
            index_add_helper<int64_t>(dim, index, data, result);
            break;
        default:
            throw std::runtime_error("invalid type");
    }
}

template <typename T>
static void gather_impl(TensorInfo<T> data, int64_t dim, TensorInfo<int64_t> index, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto index_result = result.LinearIndexToDimIndex(i);
        auto index_input  = index_result;

        index_input[dim] = index[index_result];

        result[index_result] = data[index_input];
    }
}
void gather_impl(Tensor data, int64_t dim, Tensor index, Tensor result)
{
    SWITCH_MACRO_ALL(data.scalar_type(), gather_impl, data, dim, index, result);
}


template <typename T, typename TIndex>
static void index_copy_impl(TensorInfo<T> target, int64_t dim, TensorInfo<TIndex> index, TensorInfo<T> value)
{
    for (int64_t input_linear_index = 0; input_linear_index < value.numel(); ++input_linear_index)
    {
        auto index_input = value.LinearIndexToDimIndex(input_linear_index);

        auto index_result = index_input;
        index_result[dim] = index[index_input[dim]];

        target[index_result] = value[index_input];
    }
}

template <typename TIndex>
static void index_copy_helper(Tensor& target, int64_t dim, TensorInfo<TIndex> index, Tensor value)
{
    SWITCH_MACRO_ALL(target.scalar_type(), index_copy_impl, target, dim, index, value);
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
static void transpose_impl(TensorInfo<T> input, int64_t dim0, int64_t dim1, TensorInfo<T> result)
{
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto index_result = result.LinearIndexToDimIndex(i);
        auto index_input  = index_result;
        std::swap(index_input[dim0], index_input[dim1]);
        result[index_result] = input[index_input];
    }
}

void transpose_impl(Tensor input, int64_t dim0, int64_t dim1, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), transpose_impl, input, dim0, dim1, result);
}

// ================================================================================================================



template <typename T>
static void sum_backward_impl(TensorInfo<T> grad_output, TensorInfo<T> grad_a)
{
    auto g = grad_output[0];
    for (int64_t i = 0; i < grad_a.numel(); ++i)
    {
        grad_a[i] = g;
    }
}

void sum_backward_impl(Tensor grad_output, Tensor& grad_a)
{
    SWITCH_MACRO_ALL(grad_output.scalar_type(), sum_backward_impl, grad_output, grad_a);
}


void prod_backward_impl(Tensor a, int64_t dim, Tensor grad_output, Tensor& grad_a)
{
    throw std::runtime_error("not implemented");
}

template <typename T>
static void one_backward_impl(TensorInfo<T> grad_output, TensorInfo<T> grad_a, TensorInfo<T> grad_b)
{
    for (int64_t i = 0; i < grad_output.numel(); ++i)
    {
        auto g    = grad_output[i];
        grad_a[i] = g;
        grad_b[i] = g;
    }
}

void min_backward_impl(Tensor grad_output, Tensor& grad_a, Tensor& grad_b)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl, grad_output, grad_a, grad_b);
}

void max_backward_impl(Tensor grad_output, Tensor& grad_a, Tensor& grad_b)
{
    SWITCH_MACRO_FLOAT(grad_output.scalar_type(), one_backward_impl, grad_output, grad_a, grad_b);
}


template <typename TSource, typename TTarget>
static void copy_and_convert_impl_kernel(TensorInfo<TSource> src, TensorInfo<TTarget> target)
{
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < src.numel(); ++i)
    {
        if constexpr (std::is_same_v<TTarget, Half>)
        {
            target[i] = TTarget(float(src[i]));
        }
        else
        {
            target[i] = TTarget(src[i]);
        }
    }
}

void copy_and_convert_impl(Tensor src, Tensor& target)
{
    CHECK_EQ(src.numel(), target.numel());
    if (src.dtype() == target.dtype() && src.is_contiguous() && target.is_contiguous())
    {
        memcpy(target.data_ptr(), src.data_ptr(), src.numel() * src.element_size());
        return;
    }

    switch (target.dtype())
    {
        case kUInt8:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), uint8_t, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kUInt16:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), uint16_t, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kInt16:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int16_t, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kInt32:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int32_t, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kInt64:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), int64_t, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kFloat16:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), Half, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kFloat32:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), float, copy_and_convert_impl_kernel, src, target);
            break;
        }
        case kFloat64:
        {
            SWITCH_MACRO_ALL_DUAL(src.scalar_type(), double, copy_and_convert_impl_kernel, src, target);
            break;
        }
        default:
            throw std::runtime_error("invalid type");
    }
}

template <typename T>
static void clamp_impl_(TensorInfo<T> src, double low, double high)
{
    using G        = typename CpuComputeFloatType<T>::Type;
    const G low_t  = G(std::isfinite(low) ? T(low) : std::numeric_limits<T>::lowest());
    const G high_t = G(std::isfinite(high) ? T(high) : std::numeric_limits<T>::max());
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < src.numel(); ++i)
    {
        src[i] = std::min(high_t, std::max(G(src[i]), low_t));
    }
}
void clamp_impl_(Tensor& a, double low, double high)
{
    SWITCH_MACRO_ALL(a.scalar_type(), clamp_impl_, a, low, high);
}


template <typename T>
static void repeat_interleave_impl(TensorInfo<T> input, int64_t count, TensorInfo<T> result)
{
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto index_result    = result.LinearIndexToDimIndex(i);
        auto index_input     = input.LinearIndexToDimIndex(i / count);
        result[index_result] = input[index_input];
    }
}

void repeat_interleave_impl(Tensor input, int64_t count, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), repeat_interleave_impl, input, count, result);
}



template <typename T>
static void repeat_impl(TensorInfo<T> src, TensorInfo<T> result)
{
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < result.numel(); ++i)
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
    SWITCH_MACRO_ALL(t.scalar_type(), repeat_impl, t, result);
}


}  // namespace cpu_impl
}  // namespace tinytorch