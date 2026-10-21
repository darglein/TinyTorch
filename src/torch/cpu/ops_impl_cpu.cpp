/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ops_impl_cpu.h"

#include <cmath>
#include <omp.h>
#include <vector>

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
    t = t.to(kDouble);
    print_impl<double>(strm, t);
}

void to_impl_cpu_cuda(Tensor src, Tensor dest, bool async)
{
#ifdef TT_HAS_CUDA

    CHECK_EQ(src.sizes(), dest.sizes());



    CHECK(src.is_contiguous()) << src.sizes() << " " << src.strides();
    CHECK(dest.is_contiguous()) << dest.sizes() << " " << dest.strides();

    CHECK_EQ(src.scalar_type(), dest.scalar_type());

    // cudaMemcpy2D()
    // cudaMemcpy3D();
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

    // All copies are ordered on the current streams of the involved devices. A blocking copy (async == false) waits for
    // its stream afterwards. Copies never go through the legacy default stream, so they are also correct for streams
    // that don't synchronize with it (cudaStreamNonBlocking).
    if (src.is_cpu() || dest.is_cpu())
    {
        auto type = (dest.device() == kCPU) ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        cuda::DeviceGuard guard(src.device() == kCPU ? dest.device() : src.device());
        auto strm = cuda::getCurrentCUDAStream();
        TT_CHECK_CUDA_ERROR(cudaMemcpyAsync(dest.data_ptr(), src.data_ptr(), bytes, type, strm));
        if (!async)
        {
            TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(strm));
        }
    }
    else
    {
        // p2p copy on the stream of the destination device, ordered after the work on the source stream. The source
        // stream waits for the copy, so the source buffer isn't modified before it has been read.
        bool use_uva = dest.is_uva() && src.is_uva();
        cudaEvent_t src_buffer_ready, mempcy_finished;
        {
            cuda::DeviceGuard guard(src.device());
            src_buffer_ready = cuda::getNextEvent();
            TT_CHECK_CUDA_ERROR(cudaEventRecord(src_buffer_ready, cuda::getCurrentCUDAStream()));
        }

        {
            cuda::DeviceGuard guard(dest.device());
            auto strm = cuda::getCurrentCUDAStream();
            TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(strm, src_buffer_ready));

            if (use_uva)
            {
                TT_CHECK_CUDA_ERROR(cudaMemcpyAsync(dest.data_ptr(), src.data_ptr(), bytes, cudaMemcpyDefault, strm));
            }
            else
            {
                TT_CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(dest.data_ptr(), dest.device().index(), src.data_ptr(),
                                                        src.device().index(), bytes, strm));
            }
            mempcy_finished = cuda::getNextEvent();
            TT_CHECK_CUDA_ERROR(cudaEventRecord(mempcy_finished, strm));
            if (!async)
            {
                TT_CHECK_CUDA_ERROR(cudaStreamSynchronize(strm));
            }
        }

        {
            cuda::DeviceGuard guard(src.device());
            TT_CHECK_CUDA_ERROR(cudaStreamWaitEvent(cuda::getCurrentCUDAStream(), mempcy_finished));
        }
    }


#else
    CHECK(false);
#endif
}

template <typename T>
static void sort_impl(TensorInfo<T> a, int64_t dim, TensorInfo<T> out_t, TensorInfo<int64_t> out_index)
{
    using G                = typename CpuComputeFloatType<T>::Type;
    int64_t size_along_dim = out_t.size(dim);
    if (size_along_dim == 0 || out_t.numel() == 0) return;

    // NaN sorts last (like PyTorch), equal values keep their input order
    auto less = [](const std::pair<G, int64_t>& x, const std::pair<G, int64_t>& y)
    {
        bool xn = tt_isnan(x.first);
        bool yn = tt_isnan(y.first);
        if (xn || yn)
        {
            if (xn && yn) return x.second < y.second;
            return yn;
        }
        if (x.first < y.first) return true;
        if (y.first < x.first) return false;
        return x.second < y.second;
    };

    std::vector<std::pair<G, int64_t>> data;
    data.resize(size_along_dim);

    for (int64_t i = 0; i < out_t.numel() / size_along_dim; ++i)
    {
        auto index = a.LinearIndexToDimIndexSkipDim(i, dim);
        for (int64_t o = 0; o < size_along_dim; ++o)
        {
            index[dim] = o;
            data[o]    = {G(a[index]), o};
        }
        std::sort(data.begin(), data.end(), less);
        for (int64_t o = 0; o < size_along_dim; ++o)
        {
            index[dim]       = o;
            out_t[index]     = T(data[o].first);
            out_index[index] = data[o].second;
        }
    }
}
void sort_impl(Tensor a, int64_t dim, Tensor& out_t, Tensor& out_index)
{
    dim = maybe_wrap_dim(dim, a.dim());
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
    // memset would turn -0.0 into +0.0
    if (value == 0 && !std::signbit(value) && a.is_contiguous())
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
static void rand_float_impl(TensorInfo<T> t, std::mt19937& mersenne_engine, double low, double high)
{
    using G = typename CpuComputeFloatType<T>::Type;
    auto N  = t.numel();
    for (int64_t i = 0; i < N; ++i)
    {
        T v = T(low);
        if (low < high)
        {
            // redraw in the rare case that rounding produces high, so that the result is in [low, high)
            do
            {
                double x;
                if constexpr (std::is_same<T, double>::value)
                {
                    uint64_t r = (uint64_t(mersenne_engine()) << 21) ^ uint64_t(mersenne_engine() >> 11);
                    x          = double(r & ((uint64_t(1) << 53) - 1)) * (1.0 / 9007199254740992.0);
                }
                else
                {
                    x = double(mersenne_engine() >> 8) * (1.0 / 16777216.0);
                }
                v = T(G(low + x * (high - low)));
            } while (!(double(G(v)) < high));
        }
        if (t.contiguous)
        {
            t.data[i] = v;
        }
        else
        {
            t[i] = v;
        }
    }
}

void uniform_impl(Tensor& t, double mi, double ma)
{
    SWITCH_MACRO_ALL(t.scalar_type(), rand_float_impl, t, generator(), mi, ma);
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
    CHECK_LT(low, high);
    SWITCH_MACRO_ALL(t.scalar_type(), rand_int_impl, t, generator(), low, high);
}



template <typename T>
static void normal_random_float_impl(TensorInfo<T> t, std::mt19937& mersenne_engine)
{
    std::normal_distribution<float> dist;

    auto N = t.numel();
    if (t.contiguous)
    {
        T* pt = t.data;
        for (int64_t i = 0; i < N; ++i)
        {
            float xf = dist(mersenne_engine);
            pt[i]    = T(xf);
        }
    }
    else
    {
        for (int64_t i = 0; i < N; ++i)
        {
            float xf = dist(mersenne_engine);
            t[i]     = T(xf);
        }
    }
}

void normal_random_impl(Tensor& t)
{
    SWITCH_MACRO_ALL(t.scalar_type(), normal_random_float_impl, t, generator());
}


template <typename TT>
static void add_poisson_noise_impl(TensorInfo<TT> t, std::mt19937& gen)
{
    using T = typename MathComputeType<TT>::Type;
    std::uniform_real_distribution<T> uniform_dist(0.0, 1.0);
    std::normal_distribution<T> normal_dist(0.0, 1.0);

    for (int64_t i = 0; i < t.numel(); ++i)
    {
        T lambda = T(t[i]);

        // Safety check: Photon counts cannot be negative
        if (lambda <= 0.0)
        {
            t[i] = TT(0);
            continue;
        }

        // 2. The High-Photon Fast Path (Gaussian Approximation)
        // O(1) time complexity. Used for > 99% of X-ray pixels.
        if (lambda > 15.0)
        {
            // Sample = Mean + (StdDev * Standard_Normal)
            T n      = normal_dist(gen);
            T sample = lambda + std::sqrt(lambda) * n;

            // Round to nearest integer and prevent negative photon clipping
            t[i] = TT(std::max(static_cast<T>(0.0), std::round(sample)));
        }

        // 3. The Low-Photon Accurate Path (Knuth's Method)
        // Used only for incredibly dense bone/metal where photons < 15.
        else
        {
            T L   = std::exp(-lambda);
            T p   = 1.0;
            int k = 0;

            do
            {
                k++;
                p *= uniform_dist(gen);
            } while (p > L);

            t[i] = TT(k - 1);
        }
    }
}

void add_poisson_noise_impl(Tensor& t)
{
    SWITCH_MACRO_ALL(t.scalar_type(), add_poisson_noise_impl, t, generator());
}



template <typename T, typename LoadOp>
static void global_sum_impl(TensorInfo<T> a, TensorInfo<T> result, LoadOp load_op)
{
    using A = typename CpuAccType<T>::Type;
    int n   = get_num_threads();
    std::vector<A> partial(n, A(0));

#pragma omp parallel num_threads(n)
    {
        A local_sum = A(0);

#pragma omp for schedule(static)
        for (int64_t i = 0; i < a.numel(); ++i)
        {
            A v       = A(a.contiguous ? a.data[i] : a[i]);
            local_sum = local_sum + load_op(v);
        }

        partial[omp_get_thread_num()] = local_sum;
    }

    A acc = A(result[0]);
    for (int i = 0; i < n; ++i)
    {
        acc = acc + partial[i];
    }
    result[0] = T(acc);
}

template <typename T>
static void sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    global_sum_impl(a, result, [](auto v) { return v; });
}

void sum_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sum_impl, a, result);
}

template <typename T>
static void abs_sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    global_sum_impl(a, result, [](auto v) { return v < decltype(v)(0) ? -v : v; });
}

void abs_sum_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), abs_sum_impl, a, result);
}

template <typename T>
static void prod_sum_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    global_sum_impl(a, result, [](auto v) { return v * v; });
}

void prod_sum_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), prod_sum_impl, a, result);
}

// reduces every slice along dim with op (accumulated in CpuAccType) and combines it with the value in result
template <typename T, typename Op>
static void dim_reduce_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result, Op op)
{
    using A           = typename CpuAccType<T>::Type;
    int64_t size      = input.size(dim);
    int64_t out_numel = result.numel();
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t o = 0; o < out_numel; ++o)
    {
        auto index_result = result.LinearIndexToDimIndex(o);
        auto index_input  = index_result;
        A acc             = A(result[index_result]);
        for (int64_t k = 0; k < size; ++k)
        {
            index_input[dim] = k;
            acc              = op(acc, A(input[index_input]));
        }
        result[index_result] = T(acc);
    }
}

template <typename T>
static void sum_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    dim_reduce_impl(input, dim, result, [](auto a, auto b) { return a + b; });
}

void sum_impl(Tensor a, int64_t dim, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), sum_impl, a, dim, result);
}


template <typename T>
static void prod_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    dim_reduce_impl(input, dim, result, [](auto a, auto b) { return a * b; });
}

void prod_impl(Tensor input, int64_t dim, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), prod_impl, input, dim, result);
}

template <typename T, typename Op>
static void dim_scan_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result, typename CpuAccType<T>::Type init,
                          Op op)
{
    using A      = typename CpuAccType<T>::Type;
    int64_t size = result.size(dim);
    if (size == 0 || result.numel() == 0) return;
    int64_t slices = result.numel() / size;
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t s = 0; s < slices; ++s)
    {
        auto index = result.LinearIndexToDimIndexSkipDim(s, dim);
        A acc      = init;
        for (int64_t k = 0; k < size; ++k)
        {
            index[dim]    = k;
            acc           = op(acc, A(input[index]));
            result[index] = T(acc);
        }
    }
}

template <typename T>
static void cumprod_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    dim_scan_impl(input, dim, result, typename CpuAccType<T>::Type(1), [](auto a, auto b) { return a * b; });
}
void cumprod_impl(Tensor input, int64_t dim, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), cumprod_impl, input, dim, result);
}
template <typename T>
static void cumsum_impl(TensorInfo<T> input, int64_t dim, TensorInfo<T> result)
{
    dim_scan_impl(input, dim, result, typename CpuAccType<T>::Type(0), [](auto a, auto b) { return a + b; });
}
void cumsum_impl(Tensor input, int64_t dim, Tensor result)
{
    SWITCH_MACRO_ALL(input.scalar_type(), cumsum_impl, input, dim, result);
}

// min/max that propagate NaN (the first NaN wins)
template <typename G>
static bool min_max_select(G v, G current, bool calc_min)
{
    if (tt_isnan(current)) return false;
    if (tt_isnan(v)) return true;
    return calc_min ? v < current : current < v;
}

template <typename T>
static void global_min_max_impl(TensorInfo<T> a, TensorInfo<T> result, bool calc_min)
{
    using G = typename CpuComputeFloatType<T>::Type;
    if (a.numel() == 0) return;
    G value = G(a[0]);
    for (int64_t i = 1; i < a.numel(); ++i)
    {
        G v = G(a[i]);
        if (min_max_select(v, value, calc_min))
        {
            value = v;
        }
    }
    result[0] = T(value);
}

template <typename T>
static void min_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    global_min_max_impl(a, result, true);
}

void min_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), min_impl, a, result);
}


template <typename T>
static void max_impl(TensorInfo<T> a, TensorInfo<T> result)
{
    global_min_max_impl(a, result, false);
}

void max_impl(Tensor a, Tensor result)
{
    SWITCH_MACRO_ALL(a.scalar_type(), max_impl, a, result);
}

template <typename T>
static void min_max_impl(TensorInfo<T> input, int64_t dim, TensorInfo<int64_t> indices, TensorInfo<T> result,
                         bool calc_min)
{
    using G           = typename CpuComputeFloatType<T>::Type;
    int64_t size      = input.size(dim);
    int64_t out_numel = result.numel();
    if (size == 0) return;

#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t o = 0; o < out_numel; ++o)
    {
        auto index_result = result.LinearIndexToDimIndex(o);
        auto index_input  = index_result;
        index_input[dim]  = 0;
        G best            = G(input[index_input]);
        int64_t best_k    = 0;
        for (int64_t k = 1; k < size; ++k)
        {
            index_input[dim] = k;
            G v              = G(input[index_input]);
            if (min_max_select(v, best, calc_min))
            {
                best   = v;
                best_k = k;
            }
        }
        result[index_result]  = T(best);
        indices[index_result] = best_k;
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


template <typename T>
static void median_impl(TensorInfo<T> input, TensorInfo<T, 1> result, double percentile)
{
    using G = typename CpuComputeFloatType<T>::Type;

    int64_t n = input.numel();

    if (n == 0) return;

    std::vector<G> buffer(n);
    for (int64_t i = 0; i < n; ++i)
    {
        buffer[i] = G(input[i]);
        if (tt_isnan(buffer[i]))
        {
            result[0] = T(buffer[i]);
            return;
        }
    }

    // nearest-rank percentile; for percentile 0.5 this is the lower median (like torch.median)
    int64_t k = int64_t(std::ceil(percentile * double(n))) - 1;
    k         = std::min(std::max(k, int64_t(0)), n - 1);
    std::nth_element(buffer.begin(), buffer.begin() + k, buffer.end());
    result[0] = T(buffer[k]);
}

void median_impl(Tensor a, Tensor result, double percentile)
{
    SWITCH_MACRO_ALL(a.scalar_type(), median_impl, a, result, percentile);
}

template <typename T, typename Indextype>
static void index_select_impl(TensorInfo<T> input, int64_t dim, TensorInfo<Indextype> index, TensorInfo<T> result)
{
    for (int64_t result_linear_index = 0; result_linear_index < result.numel(); ++result_linear_index)
    {
        auto index_result = result.LinearIndexToDimIndex(result_linear_index);
        auto index_input  = index_result;
        index_input[dim]  = index[index_result[dim]];
        CHECK(index_input[dim] >= 0 && index_input[dim] < input.size(dim))
            << "index_select: index " << index_input[dim] << " is out of range for dim " << dim << " with size "
            << input.size(dim);
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
        CHECK(result.index_in_range(index_result)) << "index_add: index " << index_result[dim]
                                                   << " is out of range for dim " << dim << " or data is larger than "
                                                   << "the input in another dim";

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
        CHECK(data.index_in_range(index_input)) << "gather: index " << index_input[dim] << " is out of range for dim "
                                                << dim << " or index is larger than the input in another dim";

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
        CHECK(target.index_in_range(index_result)) << "index_copy: index " << index_result[dim]
                                                   << " is out of range for dim " << dim << " or source is larger "
                                                   << "than the target in another dim";

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
    CHECK_EQ(index.dim(), 1);
    CHECK_EQ(index.numel(), value.size(dim)) << "index_copy: index and source don't match";
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
        else if constexpr ((std::is_same_v<TTarget, uint8_t> || std::is_same_v<TTarget, int16_t> ||
                            std::is_same_v<TTarget, uint16_t>) &&
                           !std::is_integral_v<TSource>)
        {
            // convert through int, which wraps negative values (a direct float -> unsigned conversion is UB)
            using F   = std::conditional_t<std::is_same_v<TSource, Half>, float, TSource>;
            target[i] = TTarget((int)F(src[i]));
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
    // a non-finite bound means unbounded (std::numeric_limits is not specialized for Half)
    using G             = typename CpuComputeFloatType<T>::Type;
    const bool has_low  = std::isfinite(low);
    const bool has_high = std::isfinite(high);
    const G low_t       = has_low ? G(clamp_bound_to<T>(low, true)) : G(0);
    const G high_t      = has_high ? G(clamp_bound_to<T>(high, false)) : G(0);
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < src.numel(); ++i)
    {
        G v = G(src[i]);
        if (has_low) v = std::max(v, low_t);
        if (has_high) v = std::min(v, high_t);
        src[i] = T(v);
    }
}
void clamp_impl_(Tensor& a, double low, double high)
{
    SWITCH_MACRO_ALL(a.scalar_type(), clamp_impl_, a, low, high);
}

template <typename T>
static void padding_2d_impl(TensorInfo<T, 4> src, TensorInfo<T, 4> dst, int pad_left, int pad_right, int pad_top,
                            int pad_bottom, PaddingMode mode)
{
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < dst.numel(); ++i)
    {
        int64_t x = i % dst.size(3);
        int64_t y = i / dst.size(3) % dst.size(2);
        int64_t c = i / dst.size(3) / dst.size(2) % dst.size(1);
        int64_t b = i / dst.size(3) / dst.size(2) / dst.size(1);

        auto x_src = x - pad_left;
        auto y_src = y - pad_top;

        if (mode == kZero)
        {
            if (x_src < 0 || x_src >= src.size(3) || y_src < 0 || y_src >= src.size(2))
            {
                dst(b, c, y, x) = T(0.);
                continue;
            }
        }
        else if (mode == kReflect)
        {
            if (x_src < 0)
            {
                x_src = -x_src;
            }
            if (y_src < 0)
            {
                y_src = -y_src;
            }

            if (x_src >= src.size(3))
            {
                x_src -= 2 * (x_src - (src.size(3) - 1));
            }
            if (y_src >= src.size(2))
            {
                y_src -= 2 * (y_src - (src.size(2) - 1));
            }
        }

        // Clamp applied for kBorder (replicate), and as a safety fallback for extreme kReflect bounds
        if (x_src < 0) x_src = 0;
        if (x_src >= src.size(3)) x_src = src.size(3) - 1;

        if (y_src < 0) y_src = 0;
        if (y_src >= src.size(2)) y_src = src.size(2) - 1;

        dst(b, c, y, x) = src(b, c, y_src, x_src);
    }
}

void padding_2d_impl(Tensor src, Tensor result, int pad_left, int pad_right, int pad_top, int pad_bottom,
                     PaddingMode mode)
{
    SWITCH_MACRO_ALL(src.scalar_type(), padding_2d_impl, src, result, pad_left, pad_right, pad_top, pad_bottom, mode);
}

template <typename T>
static void padding_3d_impl(TensorInfo<T, 5> src, TensorInfo<T, 5> dst, int pad_left, int pad_right, int pad_top,
                            int pad_bottom, int pad_front, int pad_back, PaddingMode mode)
{
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < dst.numel(); ++i)
    {
        int64_t x = i % dst.size(4);
        int64_t y = (i / dst.size(4)) % dst.size(3);
        int64_t z = (i / dst.size(4) / dst.size(3)) % dst.size(2);
        int64_t c = (i / dst.size(4) / dst.size(3) / dst.size(2)) % dst.size(1);
        int64_t b = i / dst.size(4) / dst.size(3) / dst.size(2) / dst.size(1);

        // Offset to original source coordinates
        auto x_src = x - pad_left;
        auto y_src = y - pad_top;
        auto z_src = z - pad_front;

        if (mode == kZero)
        {
            if (x_src < 0 || x_src >= src.size(4) || y_src < 0 || y_src >= src.size(3) || z_src < 0 ||
                z_src >= src.size(2))
            {
                dst(b, c, z, y, x) = T(0.);
                continue;
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
}

void padding_3d_impl(Tensor src, Tensor result, int pad_left, int pad_right, int pad_top, int pad_bottom, int pad_front,
                     int pad_back, PaddingMode mode)
{
    SWITCH_MACRO_ALL(src.scalar_type(), padding_3d_impl, src, result, pad_left, pad_right, pad_top, pad_bottom,
                     pad_front, pad_back, mode);
}
template <typename T>
static void repeat_interleave_impl(TensorInfo<T> input, int64_t count, TensorInfo<T> result)
{
#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < result.numel(); ++i)
    {
        auto index_result    = result.LinearIndexToDimIndex(i);
        auto index_input     = index_result;
        index_input[0]       = index_result[0] / count;
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

template <typename T>
static void matmul_impl_kernel(TensorInfo<T> a, TensorInfo<T> b, TensorInfo<T> result)
{
    using G = typename CpuComputeFloatType<T>::Type;

    int a_dim   = a.dim();
    int b_dim   = b.dim();
    int res_dim = result.dim();

    int64_t K = a.size(a_dim - 1);

#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t i = 0; i < result.numel(); ++i)
    {
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
}

void matmul_impl(Tensor a, Tensor b, Tensor result)
{
    // Dispatch to the correct scalar type kernel based on the tensor's dtype
    SWITCH_MACRO_ALL(a.scalar_type(), matmul_impl_kernel, a, b, result);
}

}  // namespace cpu_impl
}  // namespace tinytorch