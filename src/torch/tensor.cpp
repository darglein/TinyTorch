/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor.h"

namespace tinytorch
{


void Tensor::resize(int new_size)
{
    assert(impl_);
    impl_->data.resize(new_size, 0);
    if(impl_->autograd_meta)
    {
        impl_->autograd_meta->mutable_grad().resize(new_size);
    }
}
float& Tensor::operator[](int idx)
{
    return impl_->data[idx];
}
int Tensor::size() const
{
    return impl_->data.size();
}
Tensor::Tensor(std::vector<float> data) : impl_(std::make_shared<TensorImpl>(data)) {}
Tensor::Tensor(int size) : impl_(std::make_shared<TensorImpl>(size)) {}
void Tensor::ClearGrad()
{
    mutable_grad().impl_->data.clear();
}
const Tensor& Tensor::grad()
{
    return impl_->autograd_meta->grad();
}

Tensor& Tensor::mutable_grad()
{
    return impl_->autograd_meta->mutable_grad();
}


void Tensor::AddGradInplace(Tensor g)
{
    resize(g.size());
    for (int i = 0; i < size(); ++i)
    {
        mutable_grad().impl_->data[i] += g[i];
    }
}
void Tensor::AddInplace(Tensor g)
{
    resize(g.size());
    for (int i = 0; i < size(); ++i)
    {
        impl_->data[i] += g[i];
    }
}
std::shared_ptr<Edge> Tensor::getEdge()
{
    if (impl_->autograd_meta)
    {
        return impl_->autograd_meta->edge;
    }
    else
    {
        return nullptr;
    }
}
void Tensor::SetEdge(std::shared_ptr<Edge> edge)
{
    assert(impl_->autograd_meta);
    impl_->autograd_meta->edge = edge;
}
void Tensor::set_requires_grad(bool requires_grad)
{
    impl_->set_requires_grad(requires_grad);
}
bool Tensor::requires_grad()
{
    return impl_->requires_grad();
}

}  // namespace tinytorch