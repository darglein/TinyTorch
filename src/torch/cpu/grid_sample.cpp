/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "grid_sample.h"
#include "torch/cpu/ops_impl_cpu_helper.h"
#include "torch/core/tensor_info.h"
#include <cmath>

namespace tinytorch
{
namespace cpu_impl
{

static inline int clip_coord(int v, int limit)
{
    return std::min(limit - 1, std::max(v, 0));
}

// accumulation through a float intermediate (the CPU Half type has no arithmetic operators)
template <typename T>
static inline void accum(T& dst, T v)
{
    dst = T(float((double)dst) + float((double)v));
}

// maps normalized [-1,1] coords to pixel coords (matches cuda UVToPixel)
static inline std::pair<float, float> uv_to_pixel_2d(float u, float v, int w, int h, bool align_corners)
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

static inline std::tuple<float, float, float> uv_to_pixel_3d(float u, float v, float k, int w, int h, int d,
                                                              bool align_corners)
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

template <typename T>
static void grid_sample_2d_impl_(TensorInfo<T, 4> input, TensorInfo<T, 4> grid, InterpolationType interpolation,
                                 PaddingMode padding, bool align_corners, TensorInfo<T, 4> result)
{
    int N  = (int)input.size(0);
    int C  = (int)input.size(1);
    int IH = (int)input.size(2);
    int IW = (int)input.size(3);
    int OH = (int)grid.size(1);
    int OW = (int)grid.size(2);

#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t tid = 0; tid < (int64_t)N * OH * OW; ++tid)
    {
        int b      = (int)(tid / (OH * OW));
        int sample = (int)(tid % (OH * OW));
        int si     = sample / OW;
        int sj     = sample % OW;

        float u = (float)(double)grid(b, si, sj, 0);
        float v = (float)(double)grid(b, si, sj, 1);
        u = (u + 1) * 0.5f;
        v = (v + 1) * 0.5f;

        auto [ix, iy] = uv_to_pixel_2d(u, v, IW, IH, align_corners);

        if (interpolation == InterpolationType::kNearest)
        {
            int ix_n = clip_coord((int)std::lround(ix), IW);
            int iy_n = clip_coord((int)std::lround(iy), IH);
            for (int c = 0; c < C; ++c)
            {
                result(b, c, si, sj) = input(b, c, iy_n, ix_n);
            }
            continue;
        }

        int ix_tnw = (int)std::floor(ix);
        int iy_tnw = (int)std::floor(iy);

        int ix_tne = ix_tnw + 1, iy_tne = iy_tnw;
        int ix_tsw = ix_tnw, iy_tsw = iy_tnw + 1;
        int ix_tse = ix_tnw + 1, iy_tse = iy_tnw + 1;

        float tnw = (ix_tse - ix) * (iy_tse - iy);
        float tne = (ix - ix_tsw) * (iy_tsw - iy);
        float tsw = (ix_tne - ix) * (iy - iy_tne);
        float tse = (ix - ix_tnw) * (iy - iy_tnw);

        if (padding == PaddingMode::kZero)
        {
            if (ix_tnw < 0)
                tnw = 0, tsw = 0;
            if (ix_tne >= IW)
                tne = 0, tse = 0;
            if (iy_tnw < 0)
                tnw = 0, tne = 0;
            if (iy_tsw >= IH)
                tsw = 0, tse = 0;
        }

        int cx_tnw = clip_coord(ix_tnw, IW), cy_tnw = clip_coord(iy_tnw, IH);
        int cx_tne = clip_coord(ix_tne, IW), cy_tne = clip_coord(iy_tne, IH);
        int cx_tsw = clip_coord(ix_tsw, IW), cy_tsw = clip_coord(iy_tsw, IH);
        int cx_tse = clip_coord(ix_tse, IW), cy_tse = clip_coord(iy_tse, IH);

        for (int c = 0; c < C; ++c)
        {
            float s = 0;
            s += (float)(double)input(b, c, cy_tnw, cx_tnw) * tnw;
            s += (float)(double)input(b, c, cy_tne, cx_tne) * tne;
            s += (float)(double)input(b, c, cy_tsw, cx_tsw) * tsw;
            s += (float)(double)input(b, c, cy_tse, cx_tse) * tse;
            result(b, c, si, sj) = (T)s;
        }
    }
}

template <typename T>
static void grid_sample_2d_backward_impl_(TensorInfo<T, 4> input, TensorInfo<T, 4> grid,
                                          InterpolationType interpolation, PaddingMode padding, bool align_corners,
                                          TensorInfo<T, 4> grad_input, TensorInfo<T, 4> grad_grid,
                                          TensorInfo<T, 4> grad_result)
{
    int N  = (int)input.size(0);
    int C  = (int)input.size(1);
    int IH = (int)input.size(2);
    int IW = (int)input.size(3);
    int OH = (int)grid.size(1);
    int OW = (int)grid.size(2);

    // NOTE: deliberately serial: the scatter accumulates into shared grad_input pixels,
    // so a parallel loop over samples would race (the CUDA kernel uses atomicAdd).
    for (int64_t tid = 0; tid < (int64_t)N * OH * OW; ++tid)
    {
        int b      = (int)(tid / (OH * OW));
        int sample = (int)(tid % (OH * OW));
        int si     = sample / OW;
        int sj     = sample % OW;

        float u = (float)(double)grid(b, si, sj, 0);
        float v = (float)(double)grid(b, si, sj, 1);
        u = (u + 1) * 0.5f;
        v = (v + 1) * 0.5f;

        auto [ix, iy] = uv_to_pixel_2d(u, v, IW, IH, align_corners);

        if (interpolation == InterpolationType::kNearest)
        {
            int ix_n = clip_coord((int)std::lround(ix), IW);
            int iy_n = clip_coord((int)std::lround(iy), IH);
            for (int c = 0; c < C; ++c)
            {
                accum<T>(grad_input(b, c, iy_n, ix_n), grad_result(b, c, si, sj));
            }
            grad_grid(b, si, sj, 0) = T(0.0f);
            grad_grid(b, si, sj, 1) = T(0.0f);
            continue;
        }

        int ix_tnw = (int)std::floor(ix);
        int iy_tnw = (int)std::floor(iy);

        int ix_tne = ix_tnw + 1, iy_tne = iy_tnw;
        int ix_tsw = ix_tnw, iy_tsw = iy_tnw + 1;
        int ix_tse = ix_tnw + 1, iy_tse = iy_tnw + 1;

        float tnw = (ix_tse - ix) * (iy_tse - iy);
        float tne = (ix - ix_tsw) * (iy_tsw - iy);
        float tsw = (ix_tne - ix) * (iy - iy_tne);
        float tse = (ix - ix_tnw) * (iy - iy_tnw);

        int ix_tnw_cl = clip_coord(ix_tnw, IW), iy_tnw_cl = clip_coord(iy_tnw, IH);
        int ix_tne_cl = clip_coord(ix_tne, IW), iy_tne_cl = clip_coord(iy_tne, IH);
        int ix_tsw_cl = clip_coord(ix_tsw, IW), iy_tsw_cl = clip_coord(iy_tsw, IH);
        int ix_tse_cl = clip_coord(ix_tse, IW), iy_tse_cl = clip_coord(iy_tse, IH);

        float gix = 0;
        float giy = 0;

        for (int c = 0; c < C; ++c)
        {
            float tnw_val = (float)(double)input(b, c, iy_tnw_cl, ix_tnw_cl);
            float tne_val = (float)(double)input(b, c, iy_tne_cl, ix_tne_cl);
            float tsw_val = (float)(double)input(b, c, iy_tsw_cl, ix_tsw_cl);
            float tse_val = (float)(double)input(b, c, iy_tse_cl, ix_tse_cl);

            float g = (float)(double)grad_result(b, c, si, sj);
            accum<T>(grad_input(b, c, iy_tnw_cl, ix_tnw_cl), T(tnw * g));
            accum<T>(grad_input(b, c, iy_tne_cl, ix_tne_cl), T(tne * g));
            accum<T>(grad_input(b, c, iy_tsw_cl, ix_tsw_cl), T(tsw * g));
            accum<T>(grad_input(b, c, iy_tse_cl, ix_tse_cl), T(tse * g));

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
            gix = gix * (IW) * 0.5f;
            giy = giy * (IH) * 0.5f;
        }

        grad_grid(b, si, sj, 0) = T(gix);
        grad_grid(b, si, sj, 1) = T(giy);
    }
}

template <typename T>
static void grid_sample_3d_impl_(TensorInfo<T, 5> input, TensorInfo<T, 5> grid, InterpolationType interpolation,
                                 PaddingMode padding, bool align_corners, TensorInfo<T, 5> result)
{
    int N  = (int)input.size(0);
    int C  = (int)input.size(1);
    int ID = (int)input.size(2);
    int IH = (int)input.size(3);
    int IW = (int)input.size(4);
    int OD = (int)grid.size(1);
    int OH = (int)grid.size(2);
    int OW = (int)grid.size(3);

#pragma omp parallel for num_threads(get_num_threads())
    for (int64_t tid = 0; tid < (int64_t)N * OD * OH * OW; ++tid)
    {
        int b      = (int)(tid / (OD * OH * OW));
        int sample = (int)(tid % (OD * OH * OW));
        int sk     = sample % OW;
        int sj     = (sample / OW) % OH;
        int si     = sample / (OH * OW);

        float u = (float)(double)grid(b, si, sj, sk, 0);
        float v = (float)(double)grid(b, si, sj, sk, 1);
        float w = (float)(double)grid(b, si, sj, sk, 2);
        u = (u + 1) * 0.5f;
        v = (v + 1) * 0.5f;
        w = (w + 1) * 0.5f;

        auto [ix, iy, iz] = uv_to_pixel_3d(u, v, w, IW, IH, ID, align_corners);

        if (interpolation == InterpolationType::kNearest)
        {
            int ix_n = clip_coord((int)std::lround(ix), IW);
            int iy_n = clip_coord((int)std::lround(iy), IH);
            int iz_n = clip_coord((int)std::lround(iz), ID);
            for (int c = 0; c < C; ++c)
            {
                result(b, c, si, sj, sk) = input(b, c, iz_n, iy_n, ix_n);
            }
            continue;
        }

        int ix_tnw = (int)std::floor(ix);
        int iy_tnw = (int)std::floor(iy);
        int iz_tnw = (int)std::floor(iz);

        int ix_tne = ix_tnw + 1, iy_tne = iy_tnw, iz_tne = iz_tnw;
        int ix_tsw = ix_tnw, iy_tsw = iy_tnw + 1, iz_tsw = iz_tnw;
        int ix_tse = ix_tnw + 1, iy_tse = iy_tnw + 1, iz_tse = iz_tnw;
        int ix_bnw = ix_tnw, iy_bnw = iy_tnw, iz_bnw = iz_tnw + 1;
        int ix_bne = ix_tnw + 1, iy_bne = iy_tnw, iz_bne = iz_tnw + 1;
        int ix_bsw = ix_tnw, iy_bsw = iy_tnw + 1, iz_bsw = iz_tnw + 1;
        int ix_bse = ix_tnw + 1, iy_bse = iy_tnw + 1, iz_bse = iz_tnw + 1;

        float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
        float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
        float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
        float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
        float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
        float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
        float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
        float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

        if (padding == PaddingMode::kZero)
        {
            if (ix_tnw < 0)
                tnw = 0, tsw = 0, bnw = 0, bsw = 0;
            if (ix_tne >= IW)
                tne = 0, tse = 0, bne = 0, bse = 0;

            if (iy_tnw < 0)
                tnw = 0, tne = 0, bnw = 0, bne = 0;
            if (iy_tsw >= IH)
                tsw = 0, tse = 0, bsw = 0, bse = 0;

            if (iz_tnw < 0)
                tnw = 0, tne = 0, tsw = 0, tse = 0;
            if (iz_bnw >= ID)
                bnw = 0, bne = 0, bsw = 0, bse = 0;
        }

        int cx_tnw = clip_coord(ix_tnw, IW), cy_tnw = clip_coord(iy_tnw, IH), cz_tnw = clip_coord(iz_tnw, ID);
        int cx_tne = clip_coord(ix_tne, IW), cy_tne = clip_coord(iy_tne, IH), cz_tne = clip_coord(iz_tne, ID);
        int cx_tsw = clip_coord(ix_tsw, IW), cy_tsw = clip_coord(iy_tsw, IH), cz_tsw = clip_coord(iz_tsw, ID);
        int cx_tse = clip_coord(ix_tse, IW), cy_tse = clip_coord(iy_tse, IH), cz_tse = clip_coord(iz_tse, ID);
        int cx_bnw = clip_coord(ix_bnw, IW), cy_bnw = clip_coord(iy_bnw, IH), cz_bnw = clip_coord(iz_bnw, ID);
        int cx_bne = clip_coord(ix_bne, IW), cy_bne = clip_coord(iy_bne, IH), cz_bne = clip_coord(iz_bne, ID);
        int cx_bsw = clip_coord(ix_bsw, IW), cy_bsw = clip_coord(iy_bsw, IH), cz_bsw = clip_coord(iz_bsw, ID);
        int cx_bse = clip_coord(ix_bse, IW), cy_bse = clip_coord(iy_bse, IH), cz_bse = clip_coord(iz_bse, ID);

        for (int c = 0; c < C; ++c)
        {
            float s = 0;
            s += (float)(double)input(b, c, cz_tnw, cy_tnw, cx_tnw) * tnw;
            s += (float)(double)input(b, c, cz_tne, cy_tne, cx_tne) * tne;
            s += (float)(double)input(b, c, cz_tsw, cy_tsw, cx_tsw) * tsw;
            s += (float)(double)input(b, c, cz_tse, cy_tse, cx_tse) * tse;
            s += (float)(double)input(b, c, cz_bnw, cy_bnw, cx_bnw) * bnw;
            s += (float)(double)input(b, c, cz_bne, cy_bne, cx_bne) * bne;
            s += (float)(double)input(b, c, cz_bsw, cy_bsw, cx_bsw) * bsw;
            s += (float)(double)input(b, c, cz_bse, cy_bse, cx_bse) * bse;
            result(b, c, si, sj, sk) = (T)s;
        }
    }
}

template <typename T>
 static void grid_sample_3d_backward_impl_(TensorInfo<T, 5> input, TensorInfo<T, 5> grid,
                                           InterpolationType interpolation, PaddingMode padding, bool align_corners,
                                           TensorInfo<T, 5> grad_input, TensorInfo<T, 5> grad_grid,
                                           TensorInfo<T, 5> grad_result)
 {
     int N  = (int)input.size(0);
     int C  = (int)input.size(1);
     int ID = (int)input.size(2);
     int IH = (int)input.size(3);
     int IW = (int)input.size(4);
     int OD = (int)grid.size(1);
     int OH = (int)grid.size(2);
     int OW = (int)grid.size(3);

    // NOTE: deliberately serial: the scatter accumulates into shared grad_input pixels,
    // so a parallel loop over samples would race (the CUDA kernel uses atomicAdd).
    for (int64_t tid = 0; tid < (int64_t)N * OD * OH * OW; ++tid)
    {
        int b      = (int)(tid / (OD * OH * OW));
        int sample = (int)(tid % (OD * OH * OW));
        int sk     = sample % OW;
        int sj     = (sample / OW) % OH;
        int si     = sample / (OH * OW);

        float u = (float)(double)grid(b, si, sj, sk, 0);
        float v = (float)(double)grid(b, si, sj, sk, 1);
        float w = (float)(double)grid(b, si, sj, sk, 2);
        u = (u + 1) * 0.5f;
        v = (v + 1) * 0.5f;
        w = (w + 1) * 0.5f;

        auto [ix, iy, iz] = uv_to_pixel_3d(u, v, w, IW, IH, ID, align_corners);

        if (interpolation == InterpolationType::kNearest)
        {
            int ix_n = clip_coord((int)std::lround(ix), IW);
            int iy_n = clip_coord((int)std::lround(iy), IH);
            int iz_n = clip_coord((int)std::lround(iz), ID);
            for (int c = 0; c < C; ++c)
            {
                accum<T>(grad_input(b, c, iz_n, iy_n, ix_n), grad_result(b, c, si, sj, sk));
            }
            grad_grid(b, si, sj, sk, 0) = T(0.0f);
            grad_grid(b, si, sj, sk, 1) = T(0.0f);
            grad_grid(b, si, sj, sk, 2) = T(0.0f);
            continue;
        }

        int ix_tnw = (int)std::floor(ix);
        int iy_tnw = (int)std::floor(iy);
        int iz_tnw = (int)std::floor(iz);

        int ix_tne = ix_tnw + 1, iy_tne = iy_tnw, iz_tne = iz_tnw;
        int ix_tsw = ix_tnw, iy_tsw = iy_tnw + 1, iz_tsw = iz_tnw;
        int ix_tse = ix_tnw + 1, iy_tse = iy_tnw + 1, iz_tse = iz_tnw;
        int ix_bnw = ix_tnw, iy_bnw = iy_tnw, iz_bnw = iz_tnw + 1;
        int ix_bne = ix_tnw + 1, iy_bne = iy_tnw, iz_bne = iz_tnw + 1;
        int ix_bsw = ix_tnw, iy_bsw = iy_tnw + 1, iz_bsw = iz_tnw + 1;
        int ix_bse = ix_tnw + 1, iy_bse = iy_tnw + 1, iz_bse = iz_tnw + 1;

        float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
        float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
        float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
        float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
        float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
        float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
        float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
        float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

        int cx_tnw = clip_coord(ix_tnw, IW), cy_tnw = clip_coord(iy_tnw, IH), cz_tnw = clip_coord(iz_tnw, ID);
        int cx_tne = clip_coord(ix_tne, IW), cy_tne = clip_coord(iy_tne, IH), cz_tne = clip_coord(iz_tne, ID);
        int cx_tsw = clip_coord(ix_tsw, IW), cy_tsw = clip_coord(iy_tsw, IH), cz_tsw = clip_coord(iz_tsw, ID);
        int cx_tse = clip_coord(ix_tse, IW), cy_tse = clip_coord(iy_tse, IH), cz_tse = clip_coord(iz_tse, ID);
        int cx_bnw = clip_coord(ix_bnw, IW), cy_bnw = clip_coord(iy_bnw, IH), cz_bnw = clip_coord(iz_bnw, ID);
        int cx_bne = clip_coord(ix_bne, IW), cy_bne = clip_coord(iy_bne, IH), cz_bne = clip_coord(iz_bne, ID);
        int cx_bsw = clip_coord(ix_bsw, IW), cy_bsw = clip_coord(iy_bsw, IH), cz_bsw = clip_coord(iz_bsw, ID);
        int cx_bse = clip_coord(ix_bse, IW), cy_bse = clip_coord(iy_bse, IH), cz_bse = clip_coord(iz_bse, ID);

        float gix = 0;
        float giy = 0;
        float giz = 0;

        for (int c = 0; c < C; ++c)
        {
            float tnw_val = (float)(double)input(b, c, cz_tnw, cy_tnw, cx_tnw);
            float tne_val = (float)(double)input(b, c, cz_tne, cy_tne, cx_tne);
            float tsw_val = (float)(double)input(b, c, cz_tsw, cy_tsw, cx_tsw);
            float tse_val = (float)(double)input(b, c, cz_tse, cy_tse, cx_tse);
            float bnw_val = (float)(double)input(b, c, cz_bnw, cy_bnw, cx_bnw);
            float bne_val = (float)(double)input(b, c, cz_bne, cy_bne, cx_bne);
            float bsw_val = (float)(double)input(b, c, cz_bsw, cy_bsw, cx_bsw);
            float bse_val = (float)(double)input(b, c, cz_bse, cy_bse, cx_bse);

            float g = (float)(double)grad_result(b, c, si, sj, sk);
            accum<T>(grad_input(b, c, cz_tnw, cy_tnw, cx_tnw), T(tnw * g));
            accum<T>(grad_input(b, c, cz_tne, cy_tne, cx_tne), T(tne * g));
            accum<T>(grad_input(b, c, cz_tsw, cy_tsw, cx_tsw), T(tsw * g));
            accum<T>(grad_input(b, c, cz_tse, cy_tse, cx_tse), T(tse * g));
            accum<T>(grad_input(b, c, cz_bnw, cy_bnw, cx_bnw), T(bnw * g));
            accum<T>(grad_input(b, c, cz_bne, cy_bne, cx_bne), T(bne * g));
            accum<T>(grad_input(b, c, cz_bsw, cy_bsw, cx_bsw), T(bsw * g));
            accum<T>(grad_input(b, c, cz_bse, cy_bse, cx_bse), T(bse * g));

            float m1 = -1;
            gix += m1 * tnw_val * (iy_bse - iy) * (iz_bse - iz) * g;
            gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * g;
            gix += m1 * tsw_val * (iy - iy_bne) * (iz_bne - iz) * g;
            gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * g;
            gix += m1 * bnw_val * (iy_tse - iy) * (iz - iz_tse) * g;
            gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * g;
            gix += m1 * bsw_val * (iy - iy_tne) * (iz - iz_tne) * g;
            gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * g;

            giy += m1 * tnw_val * (ix_bse - ix) * (iz_bse - iz) * g;
            giy += m1 * tne_val * (ix - ix_bsw) * (iz_bsw - iz) * g;
            giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * g;
            giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * g;
            giy += m1 * bnw_val * (ix_tse - ix) * (iz - iz_tse) * g;
            giy += m1 * bne_val * (ix - ix_tsw) * (iz - iz_tsw) * g;
            giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * g;
            giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * g;

            giz += m1 * tnw_val * (ix_bse - ix) * (iy_bse - iy) * g;
            giz += m1 * tne_val * (ix - ix_bsw) * (iy_bsw - iy) * g;
            giz += m1 * tsw_val * (ix_bne - ix) * (iy - iy_bne) * g;
            giz += m1 * tse_val * (ix - ix_bnw) * (iy - iy_bnw) * g;
            giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * g;
            giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * g;
            giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * g;
            giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * g;
        }

        if (align_corners)
        {
            gix = gix * (IW - 1) * 0.5f;
            giy = giy * (IH - 1) * 0.5f;
            giz = giz * (ID - 1) * 0.5f;
        }
        else
        {
            gix = gix * (IW) * 0.5f;
            giy = giy * (IH) * 0.5f;
            giz = giz * (ID) * 0.5f;
        }

        grad_grid(b, si, sj, sk, 0) = T(gix);
        grad_grid(b, si, sj, sk, 1) = T(giy);
        grad_grid(b, si, sj, sk, 2) = T(giz);
    }
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
