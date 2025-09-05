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

void adam_step(Tensor param, Tensor param_grad, Tensor m1s, Tensor m2s, AdamOptions options, int& current_step)
{
    NoGradGuard ngg;
    Tensor& w = param;

    Tensor gradient = param_grad;
    Tensor m1       = m1s;
    Tensor m2       = m2s;
    float beta1     = std::get<0>(options.betas_);
    float beta2     = std::get<1>(options.betas_);
    m1              = beta1 * m1 + (1 - beta1) * gradient;
    m2              = beta2 * m2 + (1 - beta2) * (gradient * gradient);
    m1s.copy_( m1);
    m2s.copy_( m2);



    double learning_rate = options.lr();
    learning_rate *=
        std::sqrt(1.0 - std::pow(beta2, double(current_step))) / (1.0 - std::pow(beta1, double(current_step)));


    Tensor effective_learning_rate = clamp(learning_rate / (sqrt(m2 + 1e-10) + options.eps()), 0.0, 1e36);


    Tensor weight_change = effective_learning_rate * m1;
    Tensor new_weight    = w - weight_change;

    w.copy_(new_weight);
}

template <typename T>
void sgd_step(TensorInfo<T> param, TensorInfo<T> param_grad, TensorInfo<T> velocity, float momentum, float dampening,
              int step, bool nesterov, float lr)
{
    for (int i = 0; i < param.numel(); ++i)
    {
        auto& w = param[i];

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

void Adam::step()
{
    current_step++;
    for (int p = 0; p < params.size(); ++p)
    {
        auto& param = params[p];
        if (param.numel() == 0)
        {
            continue;
        }
        adam_step(param, param.mutable_grad(), m1[p], m2[p], options[p], current_step);
    }
}
void Adam::add_param_group(std::pair<std::vector<Tensor>, std::unique_ptr<AdamOptions>> group)
{
    for (auto t : group.first)
    {
        params.push_back(t);
        m1.push_back(zeros_like(t));
        m2.push_back(zeros_like(t));
        options.push_back(*group.second);
    }
}
Adam::Adam(std::vector<Tensor> params, AdamOptions options)
{
    add_param_group({params, std::make_unique<AdamOptions>(options)});
}
void SGDOptimizer::Step()
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
SGDOptimizer::SGDOptimizer(const std::vector<Tensor>& t, float lr) : lr(lr), params(t)
{
    velocities.resize(t.size());
    for (int i = 0; i < t.size(); ++i)
    {
        velocities[i] = zeros_like(t[i]);
    }
}
}  // namespace optim
}  // namespace tinytorch
