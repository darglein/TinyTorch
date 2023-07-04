/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/tensor.h"

#include "torch/core/tensor_info.h"
#include "torch/tiny_torch_config.h"

namespace tinytorch
{
namespace optim
{

struct AdamOptions
{
    typedef std::tuple<double, double> betas_t;

    AdamOptions(double lr = 1e-3) : lr_(lr) {}
    double& lr() { return lr_; }
    AdamOptions& lr(double lr)
    {
        lr_ = lr;
        return *this;
    }
    double& eps() { return eps_; }
    betas_t& betas() { return betas_; }
    void eps(double e) { eps_ = e; }
    void betas(betas_t b) { betas_ = b; }
    double& weight_decay() { return weight_decay_; }
    void weight_decay(double b) { weight_decay_ = b; }

    double lr_           = 1e-3;
    betas_t betas_       = std::make_tuple(0.9, 0.999);
    double eps_          = 1e-8;
    double weight_decay_ = 0;
    bool amsgrad_        = false;
};


struct RMSpropOptions
{
    RMSpropOptions(double lr = 1e-2) : lr_(lr) {}
    double& lr() { return lr_; }
    void eps(double e) { eps_ = e; }
    double& weight_decay() { return weight_decay_; }
    void weight_decay(double b) { weight_decay_ = b; }
    double& eps() { return eps_; }
    double& alpha() { return alpha_; }
    double lr_           = 1e-2;
    double alpha_        = 0.99;
    double eps_          = 1e-8;
    double weight_decay_ = 0;
    double momentum_     = 0;
    bool centered_       = false;
};


template <typename T>
void adam_step(TensorInfo<T> param, TensorInfo<T> param_grad, TensorInfo<T> m1s, TensorInfo<T> m2s, AdamOptions options,
               int& current_step)
{
    for (int i = 0; i < param.numel(); ++i)
    {
        auto& w = param[i];
        // assert(param.grad().size() == param.size());

        auto gradient    = param_grad[i];
        auto m1          = m1s[i];
        auto m2          = m2s[i];
        auto beta1       = std::get<0>(options.betas_);
        auto beta2       = std::get<1>(options.betas_);
        m1               = beta1 * m1 + (1 - beta1) * gradient;
        m2               = beta2 * m2 + (1 - beta2) * (gradient * gradient);
        m1s[i]           = m1;
        m2s[i]           = m2;

        auto learning_rate = options.lr();
        learning_rate *= sqrtf(1 - powf(beta2, current_step)) / (1 - powf(beta1, current_step));

        auto effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(m2) + options.eps()), 0), 1e36);

        auto weight_change = effective_learning_rate * m1;
        auto new_weight    = w - weight_change;

        w = new_weight;
    }
}

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

struct OptimizerBase
{
    int current_step = 0;
    std::vector<Tensor> params;


    void zero_grad()
    {
        for (auto& p : params)
        {
            p.mutable_grad().zero_();
        }
    }
};

struct Adam : public OptimizerBase
{
    Adam(std::vector<Tensor> params, AdamOptions options)
    {
        add_param_group({params, std::make_unique<AdamOptions>(options)});
    }

    void add_param_group(std::pair<std::vector<Tensor>, std::unique_ptr<AdamOptions>> group)
    {
        for (auto t : group.first)
        {
            params.push_back(t);
            m1.push_back(zeros_like(t));
            m2.push_back(zeros_like(t));
            options.push_back(*group.second);
        }
    }

    void step()
    {
        for (int p = 0; p < params.size(); ++p)
        {
            auto& param = params[p];
            if (param.numel() == 0)
            {
                continue;
            }
            adam_step<float>(param, param.mutable_grad(), m1[p], m2[p], options[p], current_step);
        }
        current_step++;
    }

    std::vector<AdamOptions> options;
    std::vector<Tensor> m1, m2;
};


// implemented after https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
struct SGDOptimizer : public OptimizerBase
{
    SGDOptimizer(const std::vector<Tensor>& t, float lr) : lr(lr), params(t)
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


    bool nesterov   = true;
    float dampening = 0.1f;
    int step        = 0;
    float lr;
    float momentum = 0.9f;
    std::vector<Tensor> params;
    std::vector<Tensor> velocities;
};
}  // namespace optim
}  // namespace tinytorch
