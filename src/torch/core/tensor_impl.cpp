/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "torch/core/tensor_impl.h"

#include "graph.h"
#include "torch/core/ops.h"

namespace tinytorch
{
TensorImpl::TensorImpl(const SizeType& sizes, TensorOptions options) : sizes_(sizes), options_(options)
{
    recompute_strides();
    storage_ = std::make_shared<StorageImpl>(elementSize(options.dtype_) * numel(), options.device_);


}

TensorImpl::TensorImpl(std::shared_ptr<StorageImpl> storage, int64_t storage_offset, const SizeType& sizes,
                       const SizeType& strides, TensorOptions options)
    : storage_(storage), storage_offset_(storage_offset), sizes_(sizes), strides_(strides), options_(options)
{
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

        auto grad_options = options_;
        grad_options.requires_grad(false);
        autograd_meta->_grad = zeros(sizes_, grad_options);
        autograd_meta->edge  = std::make_shared<Edge>(std::make_shared<autograd::AccumulateGrad>(Tensor(getptr())), 0);
    }
    else
    {
        autograd_meta = nullptr;
    }
}
bool TensorImpl::requires_grad() const
{
    if (!autograd_meta) return false;

    if (!autograd_meta->edge) return false;

    return true;
}

}  // namespace tinytorch