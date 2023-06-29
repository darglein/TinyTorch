/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "tensor.h"

#include "tensor_info.h"
#include "tiny_torch_config.h"

namespace tinytorch
{


template <typename T>
void sgd_step(TensorInfo<T> param, TensorInfo<T> param_grad, TensorInfo<T> velocity, float momentum, float dampening,
              int step, bool nesterov, float lr)
{
    for (int i = 0; i < param.numel(); ++i)
    {
        auto& w = param[i];
        // assert(param.grad().size() == param.size());

        auto g  = param_grad[i];
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

// implemented after https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
struct SGDOptimizer
{
    SGDOptimizer(std::vector<Tensor> t, float lr) :  lr(lr), params(t)
    {
        velocities.resize(t.size());
        for (int i = 0; i < t.size(); ++i)
        {
            velocities[i] = zeros_like(t[i]);
        }
    }

    void Step()
    {
        for (int p = 0; p < params.size(); ++p)
        {
            auto& param    = params[p];
            auto& velocity = velocities[p];
            if (param.numel() == 0)
            {
                continue;
            }
            sgd_step<float>(param, param.mutable_grad(), velocity, momentum, dampening, step, nesterov, lr);
        }
        step++;
    }

    void ZeroGrad()
    {
        for (auto& p : params)
        {
            p.mutable_grad().zero_();
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
