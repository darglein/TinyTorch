/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "tensor.h"


namespace tinytorch
{

struct SGDOptimizer
{
    SGDOptimizer(std::vector<Tensor> t, float lr) : params(t), lr(lr)
    {
        velocities.resize(t.size());
        for (int i = 0; i < t.size(); ++i)
        {
            velocities[i] = zero(t[i].size());
        }
    }

    void Step()
    {
        for (int p = 0; p < params.size(); ++p)
        {
            auto& param    = params[p];
            auto& velocity = velocities[p];
            if (param.grad().size() == 0)
            {
                continue;
            }
            for (int i = 0; i < param.size(); ++i)
            {
                auto& w = param[i];
                assert(param.grad().size() == param.size());
                auto g  = param.grad()[i];
                auto& v = velocity[i];

                // sgd with nesterov momentum
                float b;
                if (step > 0)
                {
                    b = momentum * b + (1 - 0.1) * g;

                    g = g + momentum * b;
                }

                w = w - lr * g;
            }
        }
        step++;
    }

    void ZeroGrad()
    {
        for (auto& p : params)
        {
            p.ClearGrad();
        }
    }

    int step      = 0;
    float epsilon = 1e-6;
    float lr;
    float momentum = 0.9;
    std::vector<Tensor> params;
    std::vector<Tensor> velocities;
};

}  // namespace tinytorch
