/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "grid_sample.h"
#include "torch/cuda/ops_impl_cuda_helper.h"

namespace tinytorch
{
namespace cuda_impl
{

template <typename T>
TT_HD inline std::pair<T, T> UVToPixel(T u, T v, int w, int h, bool align_corners)
{
    if (align_corners)
    {
        u = u * (w - 1);
        v = v * (h - 1);
    }
    else
    {
        u = u * w - 0.5f;
        v = v * h - 0.5f;
    }
    return {u, v};
}
template <typename T>
TT_HD inline std::tuple<T, T, T> UVToPixel(T u, T v, T k, int w, int h, int d, bool align_corners)
{
    if (align_corners)
    {
        u = u * (w - 1);
        v = v * (h - 1);
        k = k * (d - 1);
    }
    else
    {
        u = u * w - 0.5f;
        v = v * h - 0.5f;
        k = k * d - 0.5f;
    }
    return {u, v, k};
}

#undef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#undef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit - 1), MAX(in, 0))



template <typename T>
static __global__ void grid_sample_2d_impl_kernel(TensorInfoCuda<T, 4> input, TensorInfoCuda<T, 4> grid,
                                                  InterpolationType interpolation, PaddingMode padding,
                                                  bool align_corners, TensorInfoCuda<T, 4> result)
{
    int64_t b        = blockIdx.x;
    int64_t sample   = blockIdx.y * blockDim.x + threadIdx.x;
    auto num_samples = grid.size(1) * grid.size(2);
    if (sample >= num_samples) return;

    int64_t sample_i = sample / grid.size(2);
    int64_t sample_j = sample % grid.size(2);

    T u = grid(b, sample_i, sample_j, 0);
    T v = grid(b, sample_i, sample_j, 1);


    u = (u + 1) * 0.5f;
    v = (v + 1) * 0.5f;

    int IH = input.sizes[2];
    int IW = input.sizes[3];

    auto [ix, iy] = UVToPixel(u, v, IW, IH, align_corners);

    int ix_tnw = floor((ix));
    int iy_tnw = floor((iy));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;

    float tnw = (ix_tse - ix) * (iy_tse - iy);
    float tne = (ix - ix_tsw) * (iy_tsw - iy);
    float tsw = (ix_tne - ix) * (iy - iy_tne);
    float tse = (ix - ix_tnw) * (iy - iy_tnw);

    CLIP_COORDINATES(ix_tnw, ix_tnw, IW);
    CLIP_COORDINATES(iy_tnw, iy_tnw, IH);
    CLIP_COORDINATES(ix_tne, ix_tne, IW);
    CLIP_COORDINATES(iy_tne, iy_tne, IH);
    CLIP_COORDINATES(ix_tsw, ix_tsw, IW);
    CLIP_COORDINATES(iy_tsw, iy_tsw, IH);
    CLIP_COORDINATES(ix_tse, ix_tse, IW);
    CLIP_COORDINATES(iy_tse, iy_tse, IH);

    for (int c = 0; c < input.sizes[1]; ++c)
    {
        float sum = 0;
        sum += input(b, c, iy_tnw, ix_tnw) * tnw;
        sum += input(b, c, iy_tne, ix_tne) * tne;
        sum += input(b, c, iy_tsw, ix_tsw) * tsw;
        sum += input(b, c, iy_tse, ix_tse) * tse;

        result(b, c, sample_i, sample_j) = sum;
    }
}



template <typename T>
static __global__ void grid_sample_3d_impl_kernel(TensorInfoCuda<T, 5> input, TensorInfoCuda<T, 5> grid,
                                                  InterpolationType interpolation, PaddingMode padding,
                                                  bool align_corners, TensorInfoCuda<T, 5> result)
{
    int64_t b        = blockIdx.x;
    int64_t sample   = blockIdx.y * blockDim.x + threadIdx.x;
    auto num_samples = grid.size(1) * grid.size(2) * grid.size(3);
    if (sample >= num_samples) return;

    int64_t sample_k = sample % grid.size(3);
    int64_t sample_j = (sample / grid.size(3)) % grid.size(2);
    int64_t sample_i = sample / (grid.size(2) * grid.size(3));

    float u = grid(b, sample_i, sample_j, sample_k, 0);
    float v = grid(b, sample_i, sample_j, sample_k, 1);
    float w = grid(b, sample_i, sample_j, sample_k, 2);

    u = (u + 1) * 0.5f;
    v = (v + 1) * 0.5f;
    w = (w + 1) * 0.5f;

    // CUDA_KERNEL_ASSERT(u >= 0 && u <= 1);
    // CUDA_KERNEL_ASSERT(v >= 0 && v <= 1);
    // CUDA_KERNEL_ASSERT(w >= 0 && w <= 1);

    int C  = input.sizes[1];
    int ID = input.sizes[2];
    int IH = input.sizes[3];
    int IW = input.sizes[4];

    auto [ix, iy, iz] = UVToPixel(u, v, w, IW, IH, ID, align_corners);

    int ix_tnw = floor((ix));
    int iy_tnw = floor((iy));
    int iz_tnw = floor((iz));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;
    int iz_tne = iz_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;
    int iz_tsw = iz_tnw;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;
    int iz_tse = iz_tnw;

    int ix_bnw = ix_tnw;
    int iy_bnw = iy_tnw;
    int iz_bnw = iz_tnw + 1;

    int ix_bne = ix_tnw + 1;
    int iy_bne = iy_tnw;
    int iz_bne = iz_tnw + 1;

    int ix_bsw = ix_tnw;
    int iy_bsw = iy_tnw + 1;
    int iz_bsw = iz_tnw + 1;

    int ix_bse = ix_tnw + 1;
    int iy_bse = iy_tnw + 1;
    int iz_bse = iz_tnw + 1;

    float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    CLIP_COORDINATES(ix_tnw, ix_tnw, IW);
    CLIP_COORDINATES(iy_tnw, iy_tnw, IH);
    CLIP_COORDINATES(iz_tnw, iz_tnw, ID);
    CLIP_COORDINATES(ix_tne, ix_tne, IW);
    CLIP_COORDINATES(iy_tne, iy_tne, IH);
    CLIP_COORDINATES(iz_tne, iz_tne, ID);
    CLIP_COORDINATES(ix_tsw, ix_tsw, IW);
    CLIP_COORDINATES(iy_tsw, iy_tsw, IH);
    CLIP_COORDINATES(iz_tsw, iz_tsw, ID);
    CLIP_COORDINATES(ix_tse, ix_tse, IW);
    CLIP_COORDINATES(iy_tse, iy_tse, IH);
    CLIP_COORDINATES(iz_tse, iz_tse, ID);
    CLIP_COORDINATES(ix_bnw, ix_bnw, IW);
    CLIP_COORDINATES(iy_bnw, iy_bnw, IH);
    CLIP_COORDINATES(iz_bnw, iz_bnw, ID);
    CLIP_COORDINATES(ix_bne, ix_bne, IW);
    CLIP_COORDINATES(iy_bne, iy_bne, IH);
    CLIP_COORDINATES(iz_bne, iz_bne, ID);
    CLIP_COORDINATES(ix_bsw, ix_bsw, IW);
    CLIP_COORDINATES(iy_bsw, iy_bsw, IH);
    CLIP_COORDINATES(iz_bsw, iz_bsw, ID);
    CLIP_COORDINATES(ix_bse, ix_bse, IW);
    CLIP_COORDINATES(iy_bse, iy_bse, IH);
    CLIP_COORDINATES(iz_bse, iz_bse, ID);

    for (int c = 0; c < C; ++c)
    {
        float sum = 0;
        sum += float(input(b, c, iz_tnw, iy_tnw, ix_tnw)) * tnw;
        sum += float(input(b, c, iz_tne, iy_tne, ix_tne)) * tne;
        sum += float(input(b, c, iz_tsw, iy_tsw, ix_tsw)) * tsw;
        sum += float(input(b, c, iz_tse, iy_tse, ix_tse)) * tse;
        sum += float(input(b, c, iz_bnw, iy_bnw, ix_bnw)) * bnw;
        sum += float(input(b, c, iz_bne, iy_bne, ix_bne)) * bne;
        sum += float(input(b, c, iz_bsw, iy_bsw, ix_bsw)) * bsw;
        sum += float(input(b, c, iz_bse, iy_bse, ix_bse)) * bse;

        result(b, c, sample_i, sample_j, sample_k) = T(sum);
    }
}


template <typename T>
static __global__ void grid_sample_2d_backward_impl_kernel(TensorInfoCuda<T, 4> input, TensorInfoCuda<T, 4> grid,
                                                           InterpolationType interpolation, PaddingMode padding,
                                                           bool align_corners, TensorInfoCuda<T, 4> grad_input,
                                                           TensorInfoCuda<T, 4> grad_grid,
                                                           TensorInfoCuda<T, 4> grad_result)
{
    int64_t b        = blockIdx.x;
    int64_t sample   = blockIdx.y * blockDim.x + threadIdx.x;
    auto num_samples = grid.size(1) * grid.size(2);
    if (sample >= num_samples) return;

    int64_t sample_i = sample / grid.size(2);
    int64_t sample_j = sample % grid.size(2);

    T u = grid(b, sample_i, sample_j, 0);
    T v = grid(b, sample_i, sample_j, 1);


    u = (u + 1) * 0.5f;
    v = (v + 1) * 0.5f;

    int C  = input.sizes[1];
    int IH = input.sizes[2];
    int IW = input.sizes[3];

    auto [ix, iy] = UVToPixel(u, v, IW, IH, align_corners);

    int ix_tnw = floor((ix));
    int iy_tnw = floor((iy));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;

    float tnw = (ix_tse - ix) * (iy_tse - iy);
    float tne = (ix - ix_tsw) * (iy_tsw - iy);
    float tsw = (ix_tne - ix) * (iy - iy_tne);
    float tse = (ix - ix_tnw) * (iy - iy_tnw);

    int ix_tnw_cl, iy_tnw_cl, ix_tne_cl, iy_tne_cl;
    int ix_tsw_cl, iy_tsw_cl, ix_tse_cl, iy_tse_cl;

    CLIP_COORDINATES(ix_tnw, ix_tnw_cl, IW);
    CLIP_COORDINATES(iy_tnw, iy_tnw_cl, IH);
    CLIP_COORDINATES(ix_tne, ix_tne_cl, IW);
    CLIP_COORDINATES(iy_tne, iy_tne_cl, IH);
    CLIP_COORDINATES(ix_tsw, ix_tsw_cl, IW);
    CLIP_COORDINATES(iy_tsw, iy_tsw_cl, IH);
    CLIP_COORDINATES(ix_tse, ix_tse_cl, IW);
    CLIP_COORDINATES(iy_tse, iy_tse_cl, IH);

    float gix = 0;
    float giy = 0;

    for (int c = 0; c < C; ++c)
    {
        float tnw_val = input(b, c, iy_tnw_cl, ix_tnw_cl);
        float tne_val = input(b, c, iy_tne_cl, ix_tne_cl);
        float tsw_val = input(b, c, iy_tsw_cl, ix_tsw_cl);
        float tse_val = input(b, c, iy_tse_cl, ix_tse_cl);

        float g = grad_result(b, c, sample_i, sample_j);
        atomicAdd(&grad_input(b, c, iy_tnw_cl, ix_tnw_cl), tnw * g);
        atomicAdd(&grad_input(b, c, iy_tne_cl, ix_tne_cl), tne * g);
        atomicAdd(&grad_input(b, c, iy_tsw_cl, ix_tsw_cl), tsw * g);
        atomicAdd(&grad_input(b, c, iy_tse_cl, ix_tse_cl), tse * g);

        float m1 = -1;
        gix += m1 * tnw_val * (iy_tse - iy) * g;
        gix += tne_val * (iy_tsw - iy) * g;
        gix += m1 * tsw_val * (iy - iy_tne) * g;
        gix += tse_val * (iy - iy_tnw) * g;

        giy += m1 * tnw_val * (ix_tse - ix) * g;
        giy += m1 * tne_val * (ix - ix_tsw) * g;
        giy += tsw_val * (ix_tne - ix) * g;
        giy += tse_val * (ix - ix_tnw) * g;
    }

    if (align_corners)
    {
        gix = gix * (IW - 1) * 0.5f;
        giy = giy * (IH - 1) * 0.5f;
    }
    else
    {
        gix = gix * (IW)*0.5f;
        giy = giy * (IH)*0.5f;
    }


    grad_grid(b, sample_i, sample_j, 0) = gix;
    grad_grid(b, sample_i, sample_j, 1) = giy;
}


template <typename T>
static __global__ void grid_sample_3d_backward_impl_kernel(TensorInfoCuda<T, 5> input, TensorInfoCuda<T, 5> grid,
                                                           InterpolationType interpolation, PaddingMode padding,
                                                           bool align_corners, TensorInfoCuda<T, 5> grad_input,
                                                           TensorInfoCuda<T, 5> grad_grid,
                                                           TensorInfoCuda<T, 5> grad_result)
{
    int64_t b        = blockIdx.x;
    int64_t sample   = blockIdx.y * blockDim.x + threadIdx.x;
    auto num_samples = grid.size(1) * grid.size(2) * grid.size(3);
    if (sample >= num_samples) return;

    int64_t sample_k = sample % grid.size(3);
    int64_t sample_j = (sample / grid.size(3)) % grid.size(2);
    int64_t sample_i = sample / (grid.size(2) * grid.size(3));

    float u = grid(b, sample_i, sample_j, sample_k, 0);
    float v = grid(b, sample_i, sample_j, sample_k, 1);
    float w = grid(b, sample_i, sample_j, sample_k, 2);

    u = (u + 1) * 0.5f;
    v = (v + 1) * 0.5f;
    w = (w + 1) * 0.5f;

    // CUDA_KERNEL_ASSERT(u >= 0 && u <= 1);
    // CUDA_KERNEL_ASSERT(v >= 0 && v <= 1);
    // CUDA_KERNEL_ASSERT(w >= 0 && w <= 1);

    int C  = input.sizes[1];
    int ID = input.sizes[2];
    int IH = input.sizes[3];
    int IW = input.sizes[4];

    auto [ix, iy, iz] = UVToPixel(u, v, w, IW, IH, ID, align_corners);

    int ix_tnw = floor((ix));
    int iy_tnw = floor((iy));
    int iz_tnw = floor((iz));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;
    int iz_tne = iz_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;
    int iz_tsw = iz_tnw;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;
    int iz_tse = iz_tnw;

    int ix_bnw = ix_tnw;
    int iy_bnw = iy_tnw;
    int iz_bnw = iz_tnw + 1;

    int ix_bne = ix_tnw + 1;
    int iy_bne = iy_tnw;
    int iz_bne = iz_tnw + 1;

    int ix_bsw = ix_tnw;
    int iy_bsw = iy_tnw + 1;
    int iz_bsw = iz_tnw + 1;

    int ix_bse = ix_tnw + 1;
    int iy_bse = iy_tnw + 1;
    int iz_bse = iz_tnw + 1;

    float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    CLIP_COORDINATES(ix_tnw, ix_tnw, IW);
    CLIP_COORDINATES(iy_tnw, iy_tnw, IH);
    CLIP_COORDINATES(iz_tnw, iz_tnw, ID);
    CLIP_COORDINATES(ix_tne, ix_tne, IW);
    CLIP_COORDINATES(iy_tne, iy_tne, IH);
    CLIP_COORDINATES(iz_tne, iz_tne, ID);
    CLIP_COORDINATES(ix_tsw, ix_tsw, IW);
    CLIP_COORDINATES(iy_tsw, iy_tsw, IH);
    CLIP_COORDINATES(iz_tsw, iz_tsw, ID);
    CLIP_COORDINATES(ix_tse, ix_tse, IW);
    CLIP_COORDINATES(iy_tse, iy_tse, IH);
    CLIP_COORDINATES(iz_tse, iz_tse, ID);
    CLIP_COORDINATES(ix_bnw, ix_bnw, IW);
    CLIP_COORDINATES(iy_bnw, iy_bnw, IH);
    CLIP_COORDINATES(iz_bnw, iz_bnw, ID);
    CLIP_COORDINATES(ix_bne, ix_bne, IW);
    CLIP_COORDINATES(iy_bne, iy_bne, IH);
    CLIP_COORDINATES(iz_bne, iz_bne, ID);
    CLIP_COORDINATES(ix_bsw, ix_bsw, IW);
    CLIP_COORDINATES(iy_bsw, iy_bsw, IH);
    CLIP_COORDINATES(iz_bsw, iz_bsw, ID);
    CLIP_COORDINATES(ix_bse, ix_bse, IW);
    CLIP_COORDINATES(iy_bse, iy_bse, IH);
    CLIP_COORDINATES(iz_bse, iz_bse, ID);

    float gix = 0;
    float giy = 0;
    float giz = 0;

    for (int c = 0; c < input.sizes[1]; ++c)
    {
        float tnw_val = input(b, c, iz_tnw, iy_tnw, ix_tnw);
        float tne_val = input(b, c, iz_tne, iy_tne, ix_tne);
        float tsw_val = input(b, c, iz_tsw, iy_tsw, ix_tsw);
        float tse_val = input(b, c, iz_tse, iy_tse, ix_tse);
        float bnw_val = input(b, c, iz_bnw, iy_bnw, ix_bnw);
        float bne_val = input(b, c, iz_bne, iy_bne, ix_bne);
        float bsw_val = input(b, c, iz_bsw, iy_bsw, ix_bsw);
        float bse_val = input(b, c, iz_bse, iy_bse, ix_bse);

        float g = grad_result(b, c, sample_i, sample_j, sample_k);
        atomicAdd(&grad_input(b, c, iz_tnw, iy_tnw, ix_tnw), tnw * g);
        atomicAdd(&grad_input(b, c, iz_tne, iy_tne, ix_tne), tne * g);
        atomicAdd(&grad_input(b, c, iz_tsw, iy_tsw, ix_tsw), tsw * g);
        atomicAdd(&grad_input(b, c, iz_tse, iy_tse, ix_tse), tse * g);
        atomicAdd(&grad_input(b, c, iz_bnw, iy_bnw, ix_bnw), bnw * g);
        atomicAdd(&grad_input(b, c, iz_bne, iy_bne, ix_bne), bne * g);
        atomicAdd(&grad_input(b, c, iz_bsw, iy_bsw, ix_bsw), bsw * g);
        atomicAdd(&grad_input(b, c, iz_bse, iy_bse, ix_bse), bse * g);

        float gradout = g;
        float m1      = -1;
        gix += m1 * tnw_val * (iy_bse - iy) * (iz_bse - iz) * gradout;
        gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gradout;
        gix += m1 * tsw_val * (iy - iy_bne) * (iz_bne - iz) * gradout;
        gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gradout;
        gix += m1 * bnw_val * (iy_tse - iy) * (iz - iz_tse) * gradout;
        gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gradout;
        gix += m1 * bsw_val * (iy - iy_tne) * (iz - iz_tne) * gradout;
        gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gradout;


        giy += m1 * tnw_val * (ix_bse - ix) * (iz_bse - iz) * gradout;
        giy += m1 * tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gradout;
        giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gradout;
        giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gradout;
        giy += m1 * bnw_val * (ix_tse - ix) * (iz - iz_tse) * gradout;
        giy += m1 * bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gradout;
        giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gradout;
        giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gradout;

        giz += m1 * tnw_val * (ix_bse - ix) * (iy_bse - iy) * gradout;
        giz += m1 * tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gradout;
        giz += m1 * tsw_val * (ix_bne - ix) * (iy - iy_bne) * gradout;
        giz += m1 * tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gradout;
        giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gradout;
        giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gradout;
        giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gradout;
        giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gradout;
    }

    if (align_corners)
    {
        gix = gix * (IW - 1) * 0.5f;
        giy = giy * (IH - 1) * 0.5f;
        giz = giz * (ID - 1) * 0.5f;
    }
    else
    {
        gix = gix * (IW)*0.5f;
        giy = giy * (IH)*0.5f;
        giz = giz * (ID)*0.5f;
    }

    grad_grid(b, sample_i, sample_j, sample_k, 0) = gix;
    grad_grid(b, sample_i, sample_j, sample_k, 1) = giy;
    grad_grid(b, sample_i, sample_j, sample_k, 2) = giz;
}

void grid_sample_2d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{
    auto num_batches = input.size(0);
    auto num_samples = grid.size(1) * grid.size(2);
    grid_sample_2d_impl_kernel<float>
        <<<dim3(num_batches, iDivUp(num_samples, 128), 1), 128, 0, cuda::getCurrentCUDAStream()>>>(
            input, grid, interpolation, padding, align_corners, result);
    CUDA_SYNC_CHECK_ERROR();
}
void grid_sample_2d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{
    auto num_batches = input.size(0);
    auto num_samples = grid.size(1) * grid.size(2);
    grid_sample_2d_backward_impl_kernel<float>
        <<<dim3(num_batches, iDivUp(num_samples, 128), 1), 128, 0, cuda::getCurrentCUDAStream()>>>(
            input, grid, interpolation, padding, align_corners, grad_input, grad_grid, grad_result);
    CUDA_SYNC_CHECK_ERROR();
}
void grid_sample_3d_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                         bool align_corners, Tensor result)
{
    auto num_batches = input.size(0);
    auto num_samples = grid.size(1) * grid.size(2) * grid.size(3);

    switch (input.scalar_type())
    {
        case kHalf:
            grid_sample_3d_impl_kernel<__half>
                <<<dim3(num_batches, iDivUp(num_samples, 128), 1), 128, 0, cuda::getCurrentCUDAStream()>>>(
                    input, grid, interpolation, padding, align_corners, result);
            break;
        case kFloat:
            grid_sample_3d_impl_kernel<float>
                <<<dim3(num_batches, iDivUp(num_samples, 128), 1), 128, 0, cuda::getCurrentCUDAStream()>>>(
                    input, grid, interpolation, padding, align_corners, result);
            break;
        case kDouble:
            grid_sample_3d_impl_kernel<double>
                <<<dim3(num_batches, iDivUp(num_samples, 128), 1), 128, 0, cuda::getCurrentCUDAStream()>>>(
                    input, grid, interpolation, padding, align_corners, result);
            break;
        default:
            CHECK(false);
    }


    CUDA_SYNC_CHECK_ERROR();
}

void grid_sample_3d_backward_impl(Tensor input, Tensor grid, InterpolationType interpolation, PaddingMode padding,
                                  bool align_corners, Tensor& grad_input, Tensor& grad_grid, Tensor grad_result)
{
    auto num_batches = input.size(0);
    auto num_samples = grid.size(1) * grid.size(2) * grid.size(3);


    switch (input.scalar_type())
    {
        case kHalf:
            grid_sample_3d_backward_impl_kernel<__half>
                <<<dim3(num_batches, iDivUp(num_samples, 128), 1), 128, 0, cuda::getCurrentCUDAStream()>>>(
                    input, grid, interpolation, padding, align_corners, grad_input, grad_grid, grad_result);
            break;
        case kFloat:
            grid_sample_3d_backward_impl_kernel<float>
                <<<dim3(num_batches, iDivUp(num_samples, 128), 1), 128, 0, cuda::getCurrentCUDAStream()>>>(
                    input, grid, interpolation, padding, align_corners, grad_input, grad_grid, grad_result);
            break;
        default:
            CHECK(false);
    }

    CUDA_SYNC_CHECK_ERROR();
}
}  // namespace cuda_impl

}  // namespace tinytorch
