/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "grid_sample.h"
#include "torch/core/ops/grid_sample_shared.h"
#include "torch/cuda/ops_impl_cuda_helper.h"
#include <type_traits>

namespace tinytorch
{
namespace cuda_impl
{

namespace
{
struct GridSampleCudaOps
{
    template <typename A, typename T>
    __device__ static A load(T v)
    {
        if constexpr (std::is_same_v<T, __half>)
        {
            return A(__half2float(v));
        }
        else
        {
            return A(v);
        }
    }
    template <typename T, typename A>
    __device__ static void store(T& dst, A v)
    {
        if constexpr (std::is_same_v<T, __half>)
        {
            dst = __float2half(float(v));
        }
        else
        {
            dst = T(v);
        }
    }
    template <typename T, typename A>
    __device__ static void add(T* dst, A v)
    {
        if constexpr (std::is_same_v<T, __half>)
        {
            atomicAdd(dst, __float2half(float(v)));
        }
        else
        {
            atomicAdd(dst, T(v));
        }
    }
};

template <typename T>
using GridSampleAcc = std::conditional_t<std::is_same_v<T, double>, double, float>;

template <int D>
int64_t grid_sample_num_points(const Tensor& grid)
{
    int64_t n = grid.size(0);
    for (int k = 0; k < D; ++k)
    {
        n *= grid.size(1 + k);
    }
    return n;
}
}  // namespace

template <typename T, int D>
static __global__ void grid_sample_impl_kernel(TensorInfoCuda<T, D + 2> input, TensorInfoCuda<T, D + 2> grid,
                                               InterpolationType interpolation, PaddingMode padding, bool align_corners,
                                               TensorInfoCuda<T, D + 2> result, int64_t num_points)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;
    grid_sample_shared::grid_sample_forward_point<D, GridSampleAcc<T>, GridSampleCudaOps>(
        input, grid, result, tid, interpolation, padding, align_corners);
}

template <typename T, int D>
static __global__ void grid_sample_backward_impl_kernel(TensorInfoCuda<T, D + 2> input, TensorInfoCuda<T, D + 2> grid,
                                                        InterpolationType interpolation, PaddingMode padding,
                                                        bool align_corners, TensorInfoCuda<T, D + 2> grad_input,
                                                        TensorInfoCuda<T, D + 2> grad_grid,
                                                        TensorInfoCuda<T, D + 2> grad_result, int64_t num_points)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;
    grid_sample_shared::grid_sample_backward_point<D, GridSampleAcc<T>, GridSampleCudaOps>(
        input, grid, grad_input, grad_grid, grad_result, tid, interpolation, padding, align_corners);
}

template <typename T, int D>
static void grid_sample_launch(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                               bool align_corners, Tensor result)
{
    int64_t num_points = grid_sample_num_points<D>(grid);
    if (num_points == 0) return;
    grid_sample_impl_kernel<T, D><<<iDivUp(num_points, 128), 128, 0, cuda::getCurrentCUDAStream()>>>(
        input, grid, interpolation, padding, align_corners, result, num_points);
    CUDA_SYNC_CHECK_ERROR();
}

template <typename T, int D>
static void grid_sample_backward_launch(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                        bool align_corners, Tensor grad_input, Tensor grad_grid, Tensor grad_result)
{
    int64_t num_points = grid_sample_num_points<D>(grid);
    if (num_points == 0) return;
    grid_sample_backward_impl_kernel<T, D><<<iDivUp(num_points, 128), 128, 0, cuda::getCurrentCUDAStream()>>>(
        input, grid, interpolation, padding, align_corners, grad_input, grad_grid, grad_result, num_points);
    CUDA_SYNC_CHECK_ERROR();
}

#define GRID_SAMPLE_CUDA_SWITCH(scalar_type, func, D, ...)                     \
    switch (scalar_type)                                                       \
    {                                                                          \
        case kHalf:                                                            \
            func<__half, D>(__VA_ARGS__);                                      \
            break;                                                             \
        case kFloat:                                                           \
            func<float, D>(__VA_ARGS__);                                       \
            break;                                                             \
        case kDouble:                                                          \
            func<double, D>(__VA_ARGS__);                                      \
            break;                                                             \
        default:                                                               \
            CHECK(false) << "grid_sample: invalid input type " << scalar_type; \
    }

void grid_sample_2d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{
    CHECK_EQ(input.dim(), 4);
    cuda::DeviceGuard guard(input.device());
    GRID_SAMPLE_CUDA_SWITCH(input.scalar_type(), grid_sample_launch, 2, input, grid, interpolation, padding,
                            align_corners, result);
}
void grid_sample_2d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{
    CHECK_EQ(input.dim(), 4);
    cuda::DeviceGuard guard(input.device());
    GRID_SAMPLE_CUDA_SWITCH(input.scalar_type(), grid_sample_backward_launch, 2, input, grid, interpolation, padding,
                            align_corners, grad_input, grad_grid, grad_result);
}
void grid_sample_3d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{
    CHECK_EQ(input.dim(), 5);
    cuda::DeviceGuard guard(input.device());
    GRID_SAMPLE_CUDA_SWITCH(input.scalar_type(), grid_sample_launch, 3, input, grid, interpolation, padding,
                            align_corners, result);
}
void grid_sample_3d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{
    CHECK_EQ(input.dim(), 5);
    cuda::DeviceGuard guard(input.device());
    GRID_SAMPLE_CUDA_SWITCH(input.scalar_type(), grid_sample_backward_launch, 3, input, grid, interpolation, padding,
                            align_corners, grad_input, grad_grid, grad_result);
}
#undef GRID_SAMPLE_CUDA_SWITCH
}  // namespace cuda_impl

}  // namespace tinytorch
