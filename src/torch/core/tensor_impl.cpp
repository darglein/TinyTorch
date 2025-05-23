/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "torch/core/tensor_impl.h"

#include "graph.h"
#include "torch/core/ops/all.h"

namespace tinytorch
{
TensorImpl::TensorImpl(const SizeType& sizes, TensorOptions options) : sizes_(sizes), options_(options)
{
    recompute_strides();
    storage_ = std::make_shared<StorageImpl>(elementSize(options.dtype_) * numel(), options);
    options_.pinned_memory_ = storage_->options().pinned_memory_;
}

TensorImpl::TensorImpl(std::shared_ptr<StorageImpl> storage, int64_t storage_offset, const SizeType& sizes,
                       const SizeType& strides, TensorOptions options)
    : storage_(storage), storage_offset_(storage_offset), sizes_(sizes), strides_(strides), options_(options)
{
    options_.pinned_memory_ = storage_->options().pinned_memory_;
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

void TensorImpl::set_requires_grad(bool requires_grad, bool allocate_grad)
{
    if (requires_grad)
    {
        if(!this->requires_grad())
        {
            options_.requires_grad_ = true;
            autograd_meta           = std::make_unique<AutogradMeta>();
            autograd_meta->edge     = std::make_shared<Edge>(std::make_shared<autograd::AccumulateGrad>((getptr())), 0);
        }

        if (allocate_grad)
        {
            // Preallocate grad tensor
            auto options_cpy = options_;
            options_cpy.requires_grad(false);
            autograd_meta->_grad = zeros(sizes_, options_cpy);
        }
    }
    else
    {
        options_.requires_grad_ = false;
        autograd_meta           = nullptr;
    }
}
bool TensorImpl::requires_grad() const
{
    if (!autograd_meta) return false;

    if (!autograd_meta->edge) return false;

    return true;
}
void TensorImpl::set_data(TensorImpl& other)
{
    storage_        = other.storage_;
    storage_offset_ = other.storage_offset_;
    sizes_          = other.sizes_;
    strides_        = other.strides_;
    options_        = other.options_;

    bool has_grad = requires_grad() && autograd_meta->grad().defined();
    set_requires_grad(false, false);

    // recreate edges and grad tensor
    if (other.requires_grad())
    {
        set_requires_grad(true, has_grad);
    }
}

}  // namespace tinytorch