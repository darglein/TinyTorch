/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "grid_sample.h"

#include "torch/core/ops/grid_sample_shared.h"
#include "torch/core/tensor_info.h"
#include "torch/cpu/ops_impl_cpu_helper.h"
#include <type_traits>

namespace tinytorch
{
namespace cpu_impl
{

namespace
{
struct GridSampleCpuOps
{
    template <typename A, typename T>
    static A load(T v)
    {
        return A((double)v);
    }
    template <typename T, typename A>
    static void store(T& dst, A v)
    {
        dst = T(v);
    }
    template <typename T, typename A>
    static void add(T* dst, A v)
    {
        *dst = T(A((double)*dst) + v);
    }
};

template <typename T>
using GridSampleAcc = std::conditional_t<std::is_same_v<T, double>, double, float>;
}  // namespace

template <typename T, int D>
static void grid_sample_impl_(TensorInfo<T, D + 2> input, TensorInfo<T, D + 2> grid, InterpolationType interpolation,
                              PaddingMode padding, bool align_corners, TensorInfo<T, D + 2> result)
{
    int64_t num_points = grid.sizes[0];
    for (int k = 0; k < D; ++k)
    {
        num_points *= grid.sizes[1 + k];
    }

#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t tid = 0; tid < num_points; ++tid)
    {
        grid_sample_shared::grid_sample_forward_point<D, GridSampleAcc<T>, GridSampleCpuOps>(
            input, grid, result, tid, interpolation, padding, align_corners);
    }
}

template <typename T, int D>
static void grid_sample_backward_impl_(TensorInfo<T, D + 2> input, TensorInfo<T, D + 2> grid,
                                       InterpolationType interpolation, PaddingMode padding, bool align_corners,
                                       TensorInfo<T, D + 2> grad_input, TensorInfo<T, D + 2> grad_grid,
                                       TensorInfo<T, D + 2> grad_result)
{
    int64_t num_points = grid.sizes[0];
    for (int k = 0; k < D; ++k)
    {
        num_points *= grid.sizes[1 + k];
    }

    // NOTE: deliberately serial: the scatter accumulates into shared grad_input pixels,
    // so a parallel loop over samples would race (the CUDA kernel uses atomicAdd).
    for (int64_t tid = 0; tid < num_points; ++tid)
    {
        grid_sample_shared::grid_sample_backward_point<D, GridSampleAcc<T>, GridSampleCpuOps>(
            input, grid, grad_input, grad_grid, grad_result, tid, interpolation, padding, align_corners);
    }
}

template <typename T>
static void grid_sample_2d_impl_(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                 bool align_corners, Tensor result)
{
    grid_sample_impl_<T, 2>(input, grid, interpolation, padding, align_corners, result);
}
template <typename T>
static void grid_sample_3d_impl_(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                 bool align_corners, Tensor result)
{
    grid_sample_impl_<T, 3>(input, grid, interpolation, padding, align_corners, result);
}
template <typename T>
static void grid_sample_2d_backward_impl_(Tensor input, Tensor grid, InterpolationType interpolation,
                                          PaddingMode padding, bool align_corners, Tensor grad_input, Tensor grad_grid,
                                          Tensor grad_result)
{
    grid_sample_backward_impl_<T, 2>(input, grid, interpolation, padding, align_corners, grad_input, grad_grid,
                                     grad_result);
}
template <typename T>
static void grid_sample_3d_backward_impl_(Tensor input, Tensor grid, InterpolationType interpolation,
                                          PaddingMode padding, bool align_corners, Tensor grad_input, Tensor grad_grid,
                                          Tensor grad_result)
{
    grid_sample_backward_impl_<T, 3>(input, grid, interpolation, padding, align_corners, grad_input, grad_grid,
                                     grad_result);
}

void grid_sample_2d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{
    CHECK_EQ(input.dim(), 4);
    SWITCH_MACRO_FLOAT(input.scalar_type(), grid_sample_2d_impl_, input, grid, interpolation, padding, align_corners,
                       result);
}
void grid_sample_2d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{
    CHECK_EQ(input.dim(), 4);
    SWITCH_MACRO_FLOAT(input.scalar_type(), grid_sample_2d_backward_impl_, input, grid, interpolation, padding,
                       align_corners, grad_input, grad_grid, grad_result);
}
void grid_sample_3d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{
    CHECK_EQ(input.dim(), 5);
    SWITCH_MACRO_FLOAT(input.scalar_type(), grid_sample_3d_impl_, input, grid, interpolation, padding, align_corners,
                       result);
}
void grid_sample_3d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{
    CHECK_EQ(input.dim(), 5);
    SWITCH_MACRO_FLOAT(input.scalar_type(), grid_sample_3d_backward_impl_, input, grid, interpolation, padding,
                       align_corners, grad_input, grad_grid, grad_result);
}
}  // namespace cpu_impl

}  // namespace tinytorch
