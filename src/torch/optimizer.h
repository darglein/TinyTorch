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
namespace optim
{

struct AdamOptions
{
    AdamOptions(double lr = 1e-3) : lr(lr) {}
    double lr = 1e-3;
    typedef std::tuple<double, double> betas_t;
    betas_t betas       = std::make_tuple(0.9, 0.999);
    double eps          = 1e-8;
    double weight_decay = 0;
    bool amsgrad        = false;
};


struct RMSpropOptions
{
    RMSpropOptions(double lr = 1e-2) : lr(lr) {}
    double lr           = 1e-2;
    double alpha        = 0.99;
    double eps          = 1e-8;
    double weight_decay = 0;
    double momentum     = 0;
    bool centered       = false;
};


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
    SGDOptimizer(std::vector<Tensor> t, float lr) : lr(lr), params(t)
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
    float dampening = 0.1f;
    int step        = 0;
    float lr;
    float momentum = 0.5f;
    std::vector<Tensor> params;
    std::vector<Tensor> velocities;
};
}  // namespace optim
}  // namespace tinytorch
