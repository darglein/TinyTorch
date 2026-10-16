/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "reduce_helper.h"
#include "torch/core/ops/ops_impl.h"
#include "torch/cuda/atomic_minmax.h"
#include "torch/cuda/ops_impl_cuda.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include "unary_operators.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace tinytorch
{

namespace cuda_impl
{


template <typename InputType, typename OutputType, typename Op>
static __global__ void global_reduce(TensorInfoCuda<InputType> a, Op op, OutputType default_value, OutputType* partial)
{
    int64_t grid_size = blockDim.x * gridDim.x;
    int64_t num_steps = iDivUp(a.numel(), grid_size);

    OutputType value = default_value;
    for (int64_t k = 0; k < num_steps; ++k)
    {
        int64_t i              = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x + k * grid_size;
        OutputType local_value = i < a.numel() ? op.load_op(OutputType(a[i])) : default_value;
        local_value            = blockReduce<REDUCE_BLOCK_SIZE, OutputType>(local_value, op, default_value);
        value                  = op(value, local_value);
    }
    if (threadIdx.x == 0)
    {
        partial[blockIdx.x] = value;
    }
}

// Deterministic merge: a single block combines the per-block partials in a fixed
// order (no atomics), so the global result is reproducible run-to-run.
template <typename OutputType, typename Op>
static __global__ void reduce_partial(OutputType* partial, int64_t n_blocks, TensorInfoCuda<OutputType> result,
                                      Op op, OutputType default_value)
{
    int64_t num_steps = iDivUp(n_blocks, blockDim.x);
    OutputType value  = default_value;
    for (int64_t k = 0; k < num_steps; ++k)
    {
        int64_t i        = (int64_t)threadIdx.x + k * blockDim.x;
        OutputType local = i < n_blocks ? partial[i] : default_value;
        local            = blockReduce<REDUCE_BLOCK_SIZE, OutputType>(local, op, default_value);
        value            = op(value, local);
    }
    if (threadIdx.x == 0)
    {
        result[0] = value;
    }
}
template <typename InputType, typename OutputType, typename Op>
void global_reduce_launcher(TensorInfoCuda<InputType> a, Tensor result, Op op, Tensor partial, int64_t n_blocks)
{
    if (n_blocks == 0)
    {
        // empty input: nothing to reduce (result stays caller-initialized), matches prior behavior
        return;
    }
    auto stream         = cuda::getCurrentCUDAStream();
    OutputType* partial_ptr = partial.data_ptr<OutputType>();
    global_reduce<InputType, OutputType, Op>
        <<<n_blocks, REDUCE_BLOCK_SIZE, 0, stream>>>(a, op, Op::template default_value<OutputType>(), partial_ptr);
    reduce_partial<OutputType, Op>
        <<<1, REDUCE_BLOCK_SIZE, 0, stream>>>(partial_ptr, n_blocks, TensorInfoCuda<OutputType>(result), op,
                                               Op::template default_value<OutputType>());
    CUDA_SYNC_CHECK_ERROR();
}


// accumulation dtype of the global reductions: float for half, int64 for all integer types
static ScalarType global_reduce_acc_type(ScalarType t)
{
    switch (t)
    {
        case kHalf:
        case kFloat:
            return kFloat;
        case kDouble:
            return kDouble;
        default:
            return kInt64;
    }
}

template <typename Op>
void global_reduce_helper(Tensor a, Tensor result, Op op)
{
    auto acc_type      = global_reduce_acc_type(a.scalar_type());
    auto kernel_result = result;
    bool converted     = result.scalar_type() != acc_type;
    if (converted)
    {
        kernel_result = result.to(acc_type);
    }

    cuda::DeviceGuard guard(a.device());

    int64_t num_threads = std::min(int64_t(a.numel()), int64_t(1024) * 1024);
    int64_t n_blocks    = iDivUp(num_threads, REDUCE_BLOCK_SIZE);
    Tensor partial      = empty({n_blocks}, kernel_result.options());

    switch (a.scalar_type())
    {
        case kUInt8:
            global_reduce_launcher<uint8_t, int64_t, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        case kInt16:
            global_reduce_launcher<int16_t, int64_t, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        case kUInt16:
            global_reduce_launcher<uint16_t, int64_t, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        case kInt32:
            global_reduce_launcher<int, int64_t, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        case kInt64:
            global_reduce_launcher<int64_t, int64_t, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        case kFloat16:
            global_reduce_launcher<__half, float, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        case kFloat:
            global_reduce_launcher<float, float, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        case kDouble:
            global_reduce_launcher<double, double, Op>(a, kernel_result, op, partial, n_blocks);
            break;
        default:
            CHECK(false) << "invalid input type " << a.scalar_type();
    }

    if (converted)
    {
        result.copy_(kernel_result);
    }
}
void abs_sum_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceAbsAdd());
}
void prod_sum_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceProdAdd());
}
void sum_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceAdd());
}
void min_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceMin());
}
void max_impl(Tensor a, Tensor result)
{
    global_reduce_helper(a, result, ReduceMax());
}

void std_helper_impl(Tensor a, Tensor mean, Tensor result)
{
    // StdHelper reads the mean with the accumulation type of the reduction
    mean = mean.to(global_reduce_acc_type(a.scalar_type()));
    global_reduce_helper(a, result, StdHelper(mean.data_ptr()));
}


// Monotone key: smaller value <-> smaller key. NaN gets the smallest key for min and the largest key for max, so that
// it propagates like in PyTorch. 64 bit types (double, int64) use 64 bit keys, all other types 32 bit keys.
template <typename T>
struct MinMaxKey
{
    static constexpr bool is64 = sizeof(T) == 8;
    using KeyType              = typename std::conditional<is64, unsigned long long, unsigned int>::type;

    static __device__ KeyType key(T v, bool calc_min)
    {
        if constexpr (std::is_same<T, double>::value)
        {
            if (v != v) return calc_min ? 0ULL : ~0ULL;
            unsigned long long b = __double_as_longlong(v);
            return (b & 0x8000000000000000ULL) ? ~b : (b | 0x8000000000000000ULL);
        }
        else if constexpr (std::is_same<T, int64_t>::value)
        {
            return (unsigned long long)v ^ 0x8000000000000000ULL;
        }
        else if constexpr (std::is_integral<T>::value && std::is_signed<T>::value)
        {
            return (unsigned int)(int)v ^ 0x80000000u;
        }
        else if constexpr (std::is_integral<T>::value)
        {
            return (unsigned int)v;
        }
        else
        {
            float f = float(v);
            if (f != f) return calc_min ? 0u : ~0u;
            unsigned int b = __float_as_uint(f);
            return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        }
    }

    static __device__ T value(KeyType k)
    {
        if constexpr (std::is_same<T, double>::value)
        {
            unsigned long long b = (k & 0x8000000000000000ULL) ? (k ^ 0x8000000000000000ULL) : ~k;
            return __longlong_as_double(b);
        }
        else if constexpr (std::is_same<T, int64_t>::value)
        {
            return (int64_t)(k ^ 0x8000000000000000ULL);
        }
        else if constexpr (std::is_integral<T>::value && std::is_signed<T>::value)
        {
            return T((int)(k ^ 0x80000000u));
        }
        else if constexpr (std::is_integral<T>::value)
        {
            return T(k);
        }
        else
        {
            unsigned int b = (k & 0x80000000u) ? (k ^ 0x80000000u) : ~k;
            return T(__uint_as_float(b));
        }
    }
};

// scatter: one thread per input element, atomically selects (value, index) per output slot
// 32 bit keys are packed with the index into one 64 bit word. For max, the inverted index is packed, so
// that ties select the smallest index for both min and max.
template <typename T>
__launch_bounds__(128) static __global__
    void min_max_impl(TensorInfoCuda<T> input, TensorInfoCuda<T> result, int64_t dim, unsigned long long* keys,
                      bool calc_min)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= input.numel()) return;

    T v               = input[i];
    auto index_input  = input.LinearIndexToDimIndex(i);
    int64_t idx       = index_input[dim];
    auto index_result = index_input;
    index_result[dim] = 0;

    unsigned int packed_idx   = calc_min ? (unsigned int)idx : ~(unsigned int)idx;
    unsigned long long packed = ((unsigned long long)MinMaxKey<T>::key(v, calc_min) << 32) | packed_idx;
    if (calc_min)
    {
        atomicMin(keys + result.IndexToOffset(index_result), packed);
    }
    else
    {
        atomicMax(keys + result.IndexToOffset(index_result), packed);
    }
}

// gather: one thread per output slot, unpacks (key, index) into result/indices
template <typename T>
__launch_bounds__(128) static __global__
    void min_max_unpack_impl(TensorInfoCuda<T> result, TensorInfoCuda<int64_t> indices, const unsigned long long* keys,
                             bool calc_min)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    unsigned long long packed = keys[i];
    unsigned int packed_idx   = (unsigned int)(packed & 0xFFFFFFFFULL);
    result[i]                 = MinMaxKey<T>::value((unsigned int)(packed >> 32));
    indices[i]                = (int64_t)(calc_min ? packed_idx : ~packed_idx);
}

// 64 bit keys (double, int64) don't fit into one word together with the index: the first pass selects the key, the
// second the smallest index of all elements with that key.
template <typename T>
__launch_bounds__(128) static __global__ void min_max_key64_impl(TensorInfoCuda<T> input, TensorInfoCuda<T> result,
                                                                 int64_t dim, unsigned long long* keys, bool calc_min)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= input.numel()) return;

    auto index_result = input.LinearIndexToDimIndex(i);
    index_result[dim] = 0;
    auto key          = MinMaxKey<T>::key(input[i], calc_min);
    if (calc_min)
    {
        atomicMin(keys + result.IndexToOffset(index_result), key);
    }
    else
    {
        atomicMax(keys + result.IndexToOffset(index_result), key);
    }
}

template <typename T>
__launch_bounds__(128) static __global__
    void min_max_index64_impl(TensorInfoCuda<T> input, TensorInfoCuda<T> result, int64_t dim,
                              const unsigned long long* keys, unsigned long long* index, bool calc_min)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= input.numel()) return;

    auto index_input  = input.LinearIndexToDimIndex(i);
    int64_t idx       = index_input[dim];
    auto index_result = index_input;
    index_result[dim] = 0;
    auto offset       = result.IndexToOffset(index_result);
    if (MinMaxKey<T>::key(input[i], calc_min) == keys[offset])
    {
        atomicMin(index + offset, (unsigned long long)idx);
    }
}

template <typename T>
__launch_bounds__(128) static __global__
    void min_max_unpack64_impl(TensorInfoCuda<T> result, TensorInfoCuda<int64_t> indices,
                               const unsigned long long* keys, const unsigned long long* index)
{
    int64_t i = (int64_t)threadIdx.x + (int64_t)blockIdx.x * (int64_t)blockDim.x;
    if (i >= result.numel()) return;

    result[i]  = MinMaxKey<T>::value(keys[i]);
    indices[i] = (int64_t)index[i];
}

template <typename TT>
static void min_max_run(Tensor input, int64_t dim, Tensor result, Tensor indices, unsigned long long* keys_ptr,
                        int64_t numel, int64_t out_numel, bool calc_min)
{
    min_max_impl<TT><<<iDivUp(numel, 128), 128, 0, cuda::getCurrentCUDAStream()>>>(
        TensorInfoCuda<TT>(input), TensorInfoCuda<TT>(result), dim, keys_ptr, calc_min);
    min_max_unpack_impl<TT><<<iDivUp(out_numel, 128), 128, 0, cuda::getCurrentCUDAStream()>>>(
        TensorInfoCuda<TT>(result), TensorInfoCuda<int64_t>(indices), keys_ptr, calc_min);
    CUDA_SYNC_CHECK_ERROR();
}

template <typename TT>
static void min_max_run64(Tensor input, int64_t dim, Tensor result, Tensor indices, unsigned long long* keys_ptr,
                          int64_t numel, int64_t out_numel, bool calc_min)
{
    auto strm                     = cuda::getCurrentCUDAStream();
    Tensor index                  = empty(result.sizes(), TensorOptions().dtype(kLong).device(input.device()));
    unsigned long long* index_ptr = reinterpret_cast<unsigned long long*>(index.data_ptr<int64_t>());
    TT_CHECK_CUDA_ERROR(cudaMemsetAsync(index_ptr, 0xFF, sizeof(unsigned long long) * (size_t)out_numel, strm));

    min_max_key64_impl<TT><<<iDivUp(numel, 128), 128, 0, strm>>>(TensorInfoCuda<TT>(input), TensorInfoCuda<TT>(result),
                                                                 dim, keys_ptr, calc_min);
    min_max_index64_impl<TT><<<iDivUp(numel, 128), 128, 0, strm>>>(
        TensorInfoCuda<TT>(input), TensorInfoCuda<TT>(result), dim, keys_ptr, index_ptr, calc_min);
    min_max_unpack64_impl<TT><<<iDivUp(out_numel, 128), 128, 0, strm>>>(
        TensorInfoCuda<TT>(result), TensorInfoCuda<int64_t>(indices), keys_ptr, index_ptr);
    CUDA_SYNC_CHECK_ERROR();
}

static void min_max_dispatch(Tensor input, int64_t dim, Tensor result, Tensor indices, bool calc_min)
{
    int64_t out_numel = result.numel();
    int64_t numel     = input.numel();

    // scratch buffer holding the monotone keys (packed with the index for 32 bit keys)
    Tensor keys = empty(result.sizes(), TensorOptions().dtype(kLong).device(input.device()));
    unsigned long long* keys_ptr = reinterpret_cast<unsigned long long*>(keys.data_ptr<int64_t>());
    TT_CHECK_CUDA_ERROR(cudaMemsetAsync(keys_ptr, calc_min ? 0xFF : 0x00, sizeof(unsigned long long) * (size_t)out_numel,
                                        cuda::getCurrentCUDAStream()));

    switch (input.scalar_type())
    {
        case kUInt8:
            min_max_run<uint8_t>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kInt16:
            min_max_run<int16_t>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kUInt16:
            min_max_run<uint16_t>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kInt32:
            min_max_run<int32_t>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kInt64:
            min_max_run64<int64_t>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kHalf:
            min_max_run<__half>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kFloat:
            min_max_run<float>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        case kDouble:
            min_max_run64<double>(input, dim, result, indices, keys_ptr, numel, out_numel, calc_min);
            break;
        default:
            CHECK(false) << "invalid input type " << (int)input.scalar_type();
    }
}

void min_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    cuda::DeviceGuard guard(input.device());
    if (input.numel() > 0 && result.numel() > 0)
    {
        min_max_dispatch(input, dim, result, indices, true);
    }
}

void max_impl(Tensor input, int64_t dim, Tensor result, Tensor& indices)
{
    cuda::DeviceGuard guard(input.device());
    if (input.numel() > 0 && result.numel() > 0)
    {
        min_max_dispatch(input, dim, result, indices, false);
    }
}
}  // namespace cuda_impl
}  // namespace tinytorch