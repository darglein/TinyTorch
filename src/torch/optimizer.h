/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "tiny_torch_config.h"
#include "tensor.h"


namespace tinytorch
{


// implemented after https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
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
                auto& b = velocity[i];

                if (momentum != 0)
                {
                    if (step > 0)
                    {
                        b = momentum * b + (1 - dampening) * g;
                    }
                    else
                    {
                        b = g;
                    }

                    if (nesterov)
                    {
                        g = g + momentum * b;
                    }
                    else
                    {
                        g = b;
                    }
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

    bool nesterov   = true;
    float dampening = 0.1;
    int step        = 0;
    float lr;
    float momentum = 0.5;
    std::vector<Tensor> params;
    std::vector<Tensor> velocities;
};

}  // namespace tinytorch
