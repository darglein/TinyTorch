/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "optimizer.h"

#include "graph.h"
#include "torch/core/ops/all.h"

#include "torch/core/tensor_info.h"

namespace tinytorch
{
namespace optim
{


template <typename T>
void adam_step2(TensorInfo<T> param, TensorInfo<T> param_grad, TensorInfo<T> m1s, TensorInfo<T> m2s,
                AdamOptions options, int& current_step)
{
    for (int i = 0; i < param.numel(); ++i)
    {
        T& w = param[i];

        T gradient = param_grad[i];
        T m1       = m1s[i];
        T m2       = m2s[i];
        T beta1    = T(std::get<0>(options.betas_));
        T beta2    = T(std::get<1>(options.betas_));
        m1         = beta1 * m1 + (1 - beta1) * gradient;
        m2         = beta2 * m2 + (1 - beta2) * (gradient * gradient);
        m1s[i]     = m1;
        m2s[i]     = m2;

        CHECK(std::isfinite(gradient));
        CHECK(std::isfinite(m1));
        ;
        CHECK(std::isfinite(m2));
        CHECK(std::isfinite(w));

        double learning_rate = options.lr();
        learning_rate *=
            std::sqrt(1.0 - std::pow(beta2, double(current_step))) / (1.0 - std::pow(beta1, double(current_step)));
        CHECK(std::isfinite(learning_rate));

        double effective_learning_rate =
            std::min(std::max(learning_rate / (std::sqrt(m2 + 1e-10) + options.eps()), 0.0), 1e36);
        CHECK(std::isfinite(effective_learning_rate));

        double weight_change = effective_learning_rate * m1;
        double new_weight    = w - weight_change;

        CHECK(std::isfinite(new_weight));
        w = T(new_weight);
    }
}

// Moves an optimizer state tensor to the device/dtype of its parameter (e.g. after module.to()).
static void state_follow_param(Tensor& state, const Tensor& param)
{
    if (state.sizes() != param.sizes())
    {
        state = zeros_like(param);
        return;
    }
    if (!(state.device() == param.device()) || state.dtype() != param.dtype())
    {
        state = state.to(param.device()).to(param.dtype());
    }
}

static void adam_step(Tensor param, Tensor gradient, Tensor& m1s, Tensor& m2s, Tensor& m2s_max,
                      const AdamOptions& options, int current_step)
{
    NoGradGuard ngg;
    Tensor& w = param;

    CHECK_EQ(gradient.device(), param.device());
    if (gradient.dtype() != param.dtype())
    {
        gradient = gradient.to(param.dtype());
    }
    if (options.weight_decay_ != 0)
    {
        gradient = gradient + options.weight_decay_ * w;
    }

    float beta1 = std::get<0>(options.betas_);
    float beta2 = std::get<1>(options.betas_);
    Tensor m1   = beta1 * m1s + (1 - beta1) * gradient;
    Tensor m2   = beta2 * m2s + (1 - beta2) * (gradient * gradient);
    m1s.copy_(m1);
    m2s.copy_(m2);

    if (options.amsgrad_)
    {
        m2s_max.copy_(max(m2s_max, m2));
        m2 = m2s_max;
    }

    double learning_rate = options.lr_;
    learning_rate *=
        std::sqrt(1.0 - std::pow(beta2, double(current_step))) / (1.0 - std::pow(beta1, double(current_step)));


    Tensor effective_learning_rate = clamp(learning_rate / (sqrt(m2 + 1e-10) + options.eps_), 0.0, 1e36);


    Tensor weight_change = effective_learning_rate * m1;
    Tensor new_weight    = w - weight_change;

    w.copy_(new_weight);
}

void Adam::step()
{
    current_step++;
    for (int p = 0; p < params.size(); ++p)
    {
        auto& param = params[p];
        if (param.numel() == 0 || !param.grad().defined())
        {
            continue;
        }
        state_follow_param(m1[p], param);
        state_follow_param(m2[p], param);
        if (options[p].amsgrad_)
        {
            if (!m2_max[p].defined())
            {
                m2_max[p] = zeros_like(param);
            }
            state_follow_param(m2_max[p], param);
        }
        param_steps[p]++;
        adam_step(param, param.grad(), m1[p], m2[p], m2_max[p], options[p], param_steps[p]);
    }
}
void Adam::add_param_group(std::pair<std::vector<Tensor>, std::unique_ptr<AdamOptions>> group)
{
    for (auto t : group.first)
    {
        params.push_back(t);
        m1.push_back(zeros_like(t));
        m2.push_back(zeros_like(t));
        m2_max.push_back(group.second->amsgrad_ ? zeros_like(t) : Tensor());
        options.push_back(*group.second);
        param_steps.push_back(0);
    }
}
Adam::Adam(std::vector<Tensor> params, AdamOptions options)
{
    add_param_group({params, std::make_unique<AdamOptions>(options)});
}
void SGDOptimizer::Step()
{
    NoGradGuard ngg;
    for (int p = 0; p < params.size(); ++p)
    {
        auto& param = params[p];
        if (param.numel() == 0 || !param.grad().defined())
        {
            continue;
        }
        auto& velocity = velocities[p];
        state_follow_param(velocity, param);

        Tensor g = param.grad();
        CHECK_EQ(g.device(), param.device());
        if (g.dtype() != param.dtype())
        {
            g = g.to(param.dtype());
        }

        if (momentum != 0)
        {
            if (step > 0)
            {
                velocity.copy_(momentum * velocity + (1 - dampening) * g);
            }
            else
            {
                velocity.copy_(g);
            }

            if (nesterov)
            {
                g = g + momentum * velocity;
            }
            else
            {
                g = velocity;
            }
        }

        param.copy_(param - lr * g);
    }
    step++;
}
SGDOptimizer::SGDOptimizer(const std::vector<Tensor>& t, float lr) : lr(lr)
{
    params = t;  // OptimizerBase::params
    velocities.resize(t.size());
    for (int i = 0; i < t.size(); ++i)
    {
        velocities[i] = zeros_like(t[i]);
    }
}
}  // namespace optim
}  // namespace tinytorch
