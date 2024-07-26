/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "graph.h"

#include "torch/core/ops/all.h"

#include "torch/core/tensor_impl.h"
namespace tinytorch
{

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
NodeCallback* call_back = nullptr;

std::pair<Tensor, Tensor> MakeInplaceGradient(const Tensor& t, bool zero_init)
{
    if (!t.defined())
    {
        return {Tensor(), Tensor()};
    }
    if (InplaceGradientAllowed(t))
    {
        return {t.grad(), Tensor()};
    }
    else
    {
        auto g = zero_init ? zeros_like(t) : empty_like(t);
        return {g, g};
    }
}

AccumulateGrad::AccumulateGrad(std::shared_ptr<TensorImpl> t) : impl_(t)
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

    CHECK(!impl_.expired());
    auto ptr = impl_.lock();
    CHECK(ptr);

    Tensor t(ptr);

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

Node::Node()
{
    static thread_local int64_t current_seq_nr = 0;
    this->sequence_nr                          = current_seq_nr++;
}

}  // namespace autograd
}  // namespace tinytorch