/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "graph.h"
#include "torch/core/ops/all.h"
namespace tinytorch
{
int autograd::Node::current_seq_nr = 0;

static thread_local bool grad_mode_ = true;

bool GradMode::is_enabled()
{
    return grad_mode_;
}
void GradMode::set_enabled(bool enabled)
{
    grad_mode_ = enabled;
}
namespace autograd
{
AccumulateGrad::AccumulateGrad(Tensor t) : t(t)
{
    num_input_gradients_of_backward = 1;
    num_inputs_of_forward           = 0;
}
std::vector<Tensor> AccumulateGrad::node_backward(const std::vector<Tensor>& input_grad)
{
    if (num_inputs_of_forward == 0)
    {
        return {};
    }
    return input_grad;
}
std::vector<Tensor> AccumulateGrad::accumulate(const std::vector<Tensor>& input_grad)
{
    CHECK_EQ(input_grad.size(), 1);
    auto g = input_grad[0];
    if (!t.grad().defined())
    {
        t.set_grad(g);
    }
    else
    {
        t.mutable_grad() += g;
    }
    return {g};
}

}  // namespace autograd
}  // namespace tinytorch