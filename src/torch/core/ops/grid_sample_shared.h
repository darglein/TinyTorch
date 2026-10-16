/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/types.h"

#include <cmath>
#include <cstdint>

#include "torch/tiny_torch_config.h"

// Backend independent grid_sample math (PyTorch semantics). Ops provides load<A>(T), store(T&, A) and add(T*, A).
// Spatial axis a (0 = x) of the grid corresponds to input dim D + 1 - a.
namespace tinytorch
{
namespace grid_sample_shared
{

template <typename A>
TT_HD inline A clip_coordinates(A in, int64_t size, A* grad)
{
    if (in <= A(0))
    {
        *grad = A(0);
        return A(0);
    }
    A max = A(size - 1);
    if (in >= max)
    {
        *grad = A(0);
        return max;
    }
    *grad = A(1);
    return in;
}

template <typename A>
TT_HD inline A reflect_coordinates(A in, int64_t twice_low, int64_t twice_high, A* grad)
{
    if (twice_low == twice_high)
    {
        *grad = A(0);
        return A(0);
    }
    A min  = A(twice_low) / 2;
    A span = A(twice_high - twice_low) / 2;
    in     = in - min;
    A mult = A(1);
    if (in < A(0))
    {
        mult = A(-1);
        in   = -in;
    }
    A r = std::fmod(in, 2 * span);
    if (r < span)
    {
        *grad = mult;
        return r + min;
    }
    *grad = -mult;
    return 2 * span - r + min;
}

template <typename A>
TT_HD inline A compute_source_index(A coord, int64_t size, PaddingMode padding, bool align_corners, A* grad)
{
    A g_pad = A(1);
    if (align_corners)
    {
        *grad = A(size - 1) / 2;
        coord = (coord + 1) / 2 * A(size - 1);
    }
    else
    {
        *grad = A(size) / 2;
        coord = ((coord + 1) * A(size) - 1) / 2;
    }

    if (padding == kBorder)
    {
        coord = clip_coordinates(coord, size, &g_pad);
    }
    else if (padding == kReflect)
    {
        A g_ref, g_clip;
        coord = align_corners ? reflect_coordinates(coord, 0, 2 * (size - 1), &g_ref)
                              : reflect_coordinates(coord, -1, 2 * size - 1, &g_ref);
        coord = clip_coordinates(coord, size, &g_clip);
        g_pad = g_ref * g_clip;
    }
    *grad = *grad * g_pad;

    // NaN, inf and huge values are moved far outside so that the integer conversion below cannot overflow
    if (!(std::abs(coord) < A(1e9)))
    {
        coord = A(-100);
    }
    return coord;
}

template <int D, typename A, typename Ops, typename InfoT>
struct GridSamplePoint
{
    using DimIndex = typename InfoT::DimIndex;

    int64_t size[D];
    A coord[D];
    A coord_grad[D];
    DimIndex grid_index;
    DimIndex out_index;

    TT_HD GridSamplePoint(InfoT& input, InfoT& grid, int64_t tid, PaddingMode padding, bool align_corners)
    {
        int64_t rem = tid;
        for (int k = D - 1; k >= 0; --k)
        {
            int64_t s         = grid.sizes[1 + k];
            int64_t p         = rem % s;
            rem               = rem / s;
            grid_index[1 + k] = p;
            out_index[2 + k]  = p;
        }
        grid_index[0] = rem;
        out_index[0]  = rem;
        out_index[1]  = 0;
        for (int a = 0; a < D; ++a)
        {
            size[a] = input.sizes[D + 1 - a];
        }
        for (int a = 0; a < D; ++a)
        {
            grid_index[D + 1] = a;
            A g               = Ops::template load<A>(grid[grid_index]);
            coord[a]          = compute_source_index(g, size[a], padding, align_corners, &coord_grad[a]);
        }
    }
};

template <typename Ops, typename A, typename T>
TT_HD inline A load(T v)
{
    return Ops::template load<A>(v);
}

template <int D, typename A, typename Ops, typename InfoT>
TT_HD inline void grid_sample_forward_point(InfoT& input, InfoT& grid, InfoT& result, int64_t tid,
                                            InterpolationType interpolation, PaddingMode padding, bool align_corners)
{
    using DimIndex   = typename InfoT::DimIndex;
    constexpr int NC = 1 << D;

    GridSamplePoint<D, A, Ops, InfoT> pt(input, grid, tid, padding, align_corners);
    int64_t C        = input.sizes[1];
    DimIndex out_idx = pt.out_index;
    DimIndex in_idx;
    in_idx[0] = out_idx[0];

    if (interpolation == kNearest)
    {
        bool inside = true;
        for (int a = 0; a < D; ++a)
        {
            int64_t p         = (int64_t)std::nearbyint(pt.coord[a]);
            inside            = inside && p >= 0 && p < pt.size[a];
            in_idx[D + 1 - a] = p;
        }
        for (int64_t c = 0; c < C; ++c)
        {
            out_idx[1] = c;
            in_idx[1]  = c;
            Ops::store(result[out_idx], inside ? load<Ops, A>(input[in_idx]) : A(0));
        }
        return;
    }

    int64_t base[D];
    A frac[D];
    for (int a = 0; a < D; ++a)
    {
        base[a] = (int64_t)std::floor(pt.coord[a]);
        frac[a] = pt.coord[a] - A(base[a]);
    }

    A weight[NC];
    bool inside[NC];
    for (int corner = 0; corner < NC; ++corner)
    {
        weight[corner] = A(1);
        inside[corner] = true;
        for (int a = 0; a < D; ++a)
        {
            int bit        = (corner >> a) & 1;
            int64_t p      = base[a] + bit;
            inside[corner] = inside[corner] && p >= 0 && p < pt.size[a];
            weight[corner] = weight[corner] * (bit ? frac[a] : A(1) - frac[a]);
        }
    }

    for (int64_t c = 0; c < C; ++c)
    {
        in_idx[1] = c;
        A sum     = A(0);
        for (int corner = 0; corner < NC; ++corner)
        {
            if (!inside[corner])
            {
                continue;
            }
            for (int a = 0; a < D; ++a)
            {
                in_idx[D + 1 - a] = base[a] + ((corner >> a) & 1);
            }
            sum = sum + weight[corner] * load<Ops, A>(input[in_idx]);
        }
        out_idx[1] = c;
        Ops::store(result[out_idx], sum);
    }
}

template <int D, typename A, typename Ops, typename InfoT>
TT_HD inline void grid_sample_backward_point(InfoT& input, InfoT& grid, InfoT& grad_input, InfoT& grad_grid,
                                             InfoT& grad_result, int64_t tid, InterpolationType interpolation,
                                             PaddingMode padding, bool align_corners)
{
    using DimIndex   = typename InfoT::DimIndex;
    constexpr int NC = 1 << D;

    GridSamplePoint<D, A, Ops, InfoT> pt(input, grid, tid, padding, align_corners);
    int64_t C        = input.sizes[1];
    DimIndex out_idx = pt.out_index;
    DimIndex in_idx;
    in_idx[0]         = out_idx[0];
    DimIndex grid_idx = pt.grid_index;

    if (interpolation == kNearest)
    {
        bool inside = true;
        for (int a = 0; a < D; ++a)
        {
            int64_t p         = (int64_t)std::nearbyint(pt.coord[a]);
            inside            = inside && p >= 0 && p < pt.size[a];
            in_idx[D + 1 - a] = p;
        }
        if (inside)
        {
            for (int64_t c = 0; c < C; ++c)
            {
                out_idx[1] = c;
                in_idx[1]  = c;
                Ops::add(&grad_input[in_idx], load<Ops, A>(grad_result[out_idx]));
            }
        }
        for (int a = 0; a < D; ++a)
        {
            grid_idx[D + 1] = a;
            Ops::store(grad_grid[grid_idx], A(0));
        }
        return;
    }

    int64_t base[D];
    A frac[D];
    for (int a = 0; a < D; ++a)
    {
        base[a] = (int64_t)std::floor(pt.coord[a]);
        frac[a] = pt.coord[a] - A(base[a]);
    }

    A weight[NC];
    A dweight[NC][D];
    bool inside[NC];
    for (int corner = 0; corner < NC; ++corner)
    {
        weight[corner] = A(1);
        inside[corner] = true;
        for (int a = 0; a < D; ++a)
        {
            dweight[corner][a] = A(1);
        }
        for (int a = 0; a < D; ++a)
        {
            int bit        = (corner >> a) & 1;
            int64_t p      = base[a] + bit;
            inside[corner] = inside[corner] && p >= 0 && p < pt.size[a];
            A w            = bit ? frac[a] : A(1) - frac[a];
            weight[corner] = weight[corner] * w;
            for (int e = 0; e < D; ++e)
            {
                dweight[corner][e] = dweight[corner][e] * (e == a ? (bit ? A(1) : A(-1)) : w);
            }
        }
    }

    A gcoord[D];
    for (int a = 0; a < D; ++a)
    {
        gcoord[a] = A(0);
    }

    for (int64_t c = 0; c < C; ++c)
    {
        in_idx[1]  = c;
        out_idx[1] = c;
        A g        = load<Ops, A>(grad_result[out_idx]);
        for (int corner = 0; corner < NC; ++corner)
        {
            if (!inside[corner])
            {
                continue;
            }
            for (int a = 0; a < D; ++a)
            {
                in_idx[D + 1 - a] = base[a] + ((corner >> a) & 1);
            }
            Ops::add(&grad_input[in_idx], weight[corner] * g);
            A val = load<Ops, A>(input[in_idx]);
            for (int a = 0; a < D; ++a)
            {
                gcoord[a] = gcoord[a] + val * dweight[corner][a] * g;
            }
        }
    }

    for (int a = 0; a < D; ++a)
    {
        grid_idx[D + 1] = a;
        Ops::store(grad_grid[grid_idx], gcoord[a] * pt.coord_grad[a]);
    }
}

}  // namespace grid_sample_shared
}  // namespace tinytorch
