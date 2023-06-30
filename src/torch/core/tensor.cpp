/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "torch/core/tensor.h"

#include "torch/core/ops.h"

namespace TINY_TORCH_NAMESPACE
{

int64_t Tensor::numel() const
{
    return impl_->numel();
}

const Tensor& Tensor::grad() const
{
    return impl_->autograd_meta->grad();
}

Tensor& Tensor::mutable_grad()
{
    return impl_->autograd_meta->mutable_grad();
}

std::shared_ptr<Edge> Tensor::getEdge() const
{
    return impl_->autograd_meta ? impl_->autograd_meta->edge : nullptr;
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
bool Tensor::requires_grad() const
{
    return impl_->requires_grad();
}
int64_t Tensor::element_size() const
{
    return elementSize(dtype());
}
Device Tensor::device() const
{
    return impl_->options_.device_;
}
uint8_t* Tensor::ptr() const
{
    return impl_->data_ptr();
}
const SizeType& Tensor::strides() const
{
    return impl_->strides_;
}
const SizeType& Tensor::sizes() const
{
    return impl_->sizes_;
}
int64_t Tensor::dim() const
{
    return impl_->dim();
}
int64_t Tensor::size(int64_t index) const
{
    return impl_->sizes_[index];
}
int64_t Tensor::stride(int64_t index) const
{
    return impl_->strides_[index];
}
void Tensor::zero_()
{
    fill(*this, 0);
}
ScalarType Tensor::scalar_type() const
{
    return impl_->options_.dtype_;
}
TensorOptions Tensor::options() const
{
    return impl_->options_;
}

Tensor Tensor::squeeze(int64_t dim) const 
{
    assert(size(dim) == 1);

    std::vector<int64_t> new_sizes = sizes().vec();
    new_sizes.erase(std::next(new_sizes.begin(), dim));
    std::shared_ptr<TensorImpl> new_impl = std::make_shared<TensorImpl>(impl_->storage_, new_sizes, options());
    return Tensor(new_impl);
}



TensorImpl::TensorImpl(const SizeType& sizes, TensorOptions options) : sizes_(sizes), options_(options)
{
    recompute_strides();
    storage_ = std::make_shared<StorageImpl>(elementSize(options.dtype_) * numel(), kCPU);
}

TensorImpl::TensorImpl(std::shared_ptr<StorageImpl> storage, const SizeType& sizes, TensorOptions options)
    : storage_(storage), sizes_(sizes), options_(options)
{
    recompute_strides();
}

void TensorImpl::recompute_strides() 
{
    strides_.resize(dim());
    int64_t stride = 1;
    for (int64_t i = dim() - 1; i >= 0; --i)
    {
        strides_[i] = stride;
        stride *= sizes_[i];
    }
}

void TensorImpl::set_requires_grad(bool requires_grad)
{
    if (requires_grad)
    {
        autograd_meta        = std::make_unique<AutogradMeta>();
        autograd_meta->_grad = zeros(sizes_);
    }
    else
    {
        autograd_meta = nullptr;
    }
}
}  // namespace TINY_TORCH_NAMESPACE