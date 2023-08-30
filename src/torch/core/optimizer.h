/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/tensor.h"

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



struct OptimizerBase
{
    int current_step = 0;
    std::vector<Tensor> params;


    void zero_grad()
    {
        for (auto& p : params)
        {
            if(p.grad().defined()) p.mutable_grad().zero_();
        }
    }
};

struct TINYTORCH_API Adam : public OptimizerBase
{
    Adam(std::vector<Tensor> params, AdamOptions options);

    void add_param_group(std::pair<std::vector<Tensor>, std::unique_ptr<AdamOptions>> group);

    void step();

    std::vector<AdamOptions> options;
    std::vector<Tensor> m1, m2;
};


// implemented after https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
struct TINYTORCH_API SGDOptimizer : public OptimizerBase
{
    SGDOptimizer(const std::vector<Tensor>& t, float lr);

    void Step();


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
