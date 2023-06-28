/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "tensor.h"

#include "ops.h"
namespace tinytorch
{


void Tensor::resize(int new_size)
{
    //    assert(impl_);
    //    impl_->data.resize(new_size, 0);
    //    if (impl_->autograd_meta)
    //    {
    //        impl_->autograd_meta->mutable_grad().resize(new_size);
    //    }
}
// float& Tensor::operator[](int idx)
//{
//     return impl_->data[idx];
// }
int64_t Tensor::numel() const
{
    return impl_->numel();
}
// Tensor::Tensor(std::vector<float> data) : impl_(std::make_shared<TensorImpl>(data)) {}
// Tensor::Tensor(int size) : impl_(std::make_shared<TensorImpl>(size)) {}
// void Tensor::ClearGrad()
//{
//    mutable_grad().impl_->data.clear();
//}
const Tensor& Tensor::grad()
{
    return impl_->autograd_meta->grad();
}

Tensor& Tensor::mutable_grad()
{
    return impl_->autograd_meta->mutable_grad();
}


// void Tensor::AddGradInplace(Tensor g)
//{
//     resize(g.size());
//     for (int i = 0; i < size(); ++i)
//     {
//         mutable_grad().impl_->data[i] += g[i];
//     }
// }
// void Tensor::AddInplace(Tensor g)
//{
//     resize(g.size());
//     for (int i = 0; i < size(); ++i)
//     {
//         impl_->data[i] += g[i];
//     }
// }
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
uint8_t* Tensor::ptr()
{
    return impl_->ptr();
}
const std::vector<int64_t>& Tensor::sizes() const
{
    return impl_->sizes_;
}
int64_t Tensor::dim()
{
    return impl_->dim();
}
int64_t Tensor::size(int64_t index)
{
    return impl_->sizes_[index];
}
int64_t Tensor::stride(int64_t index)
{
    return impl_->strides_[index];
}

void Tensor::zero_()
{
    fill_impl(*this, 0);
}


TensorImpl::TensorImpl(std::vector<int64_t> sizes, ScalarType scalar_type) : sizes_(sizes), scalar_type_(scalar_type)
{
    int64_t bytes_per_element = 4;

    strides_.resize(dim());
    int64_t stride = 1;
    for (int64_t i = dim() - 1; i >= 0; --i)
    {
        strides_[i] = stride;
        stride *= sizes[i];
    }
    storage_ = std::make_shared<StorageImpl>(bytes_per_element * numel(), kCPU);
}
void TensorImpl::set_requires_grad(bool requires_grad)
{
    if (requires_grad)
    {
        autograd_meta = std::make_unique<AutogradMeta>();
        autograd_meta->_grad = zeros(sizes_);
    }
    else
    {
        autograd_meta = nullptr;
    }
}
}  // namespace tinytorch