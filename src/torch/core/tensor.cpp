#include "tensor.h"
/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "torch/core/tensor.h"

#include "torch/core/backward.h"
#include "torch/core/ops.h"

#include "torch/core/tensor_impl.h"

namespace tinytorch
{
int64_t Tensor::numel() const
{
    return impl_->numel();
}

Tensor Tensor::grad() const
{
    if (!impl_->autograd_meta)
    {
        return {};
    }
    return impl_->autograd_meta->grad();
}

Tensor& Tensor::mutable_grad()
{
    CHECK(impl_->autograd_meta);
    return impl_->autograd_meta->mutable_grad();
}

std::shared_ptr<Edge> Tensor::getEdge() const
{
    return impl_->autograd_meta ? impl_->autograd_meta->edge : nullptr;
}
void Tensor::SetEdge(std::shared_ptr<Edge> edge)
{
    CHECK(impl_->autograd_meta);
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

static void fill_neg_one_dim(SizeType& new_sizes, int64_t old_numel)
{
    int64_t new_numel = 1;
    int64_t* neg_dim  = nullptr;
    for (int64_t& i : new_sizes.vec())
    {
        if (i == -1)
        {
            CHECK_EQ(neg_dim, nullptr);
            neg_dim = &i;
        }
        else
        {
            new_numel *= i;
        }
    }

    if (neg_dim)
    {
        CHECK_EQ(old_numel % new_numel, 0);
        *neg_dim = old_numel / new_numel;
        new_numel *= *neg_dim;
    }

    CHECK_EQ(old_numel, new_numel);
}

Tensor Tensor::view(const SizeType& sizes) const
{
    SizeType new_sizes = sizes;
    fill_neg_one_dim(new_sizes, numel());

    if (is_contiguous())
    {
        SizeType new_strides;
        new_strides.resize(new_sizes.vec().size());
        int64_t stride = 1;
        for (int64_t i = new_strides.vec().size() - 1; i >= 0; --i)
        {
            new_strides[i] = stride;
            stride *= new_sizes[i];
        }

        std::shared_ptr<TensorImpl> new_impl = std::make_shared<TensorImpl>(
            impl_->storage_, impl_->storage_offset_, std::move(new_sizes), std::move(new_strides), options());

        return Tensor(new_impl);
    }

    throw std::runtime_error("not implemented");

    return {};
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const
{
    int64_t dims = this->dim();

    CHECK_LT(dim, dims);
    CHECK_LT(start, end);
    CHECK_LE(end, size(dim));
    CHECK_EQ((end - start) % step, 0);

    int64_t offset = start * stride(dim);
    offset *= element_size();

    std::vector<int64_t> new_sizes = sizes();
    new_sizes[dim]                 = (end - start) / step;

    auto new_strides = strides();
    new_strides[dim] *= step;

    std::shared_ptr<TensorImpl> new_impl = std::make_shared<TensorImpl>(
        impl_->storage_, impl_->storage_offset_ + offset, std::move(new_sizes), std::move(new_strides), options());

    return Tensor(new_impl);
}

Tensor Tensor::unsqueeze(int64_t dim) const
{
    CHECK(dim >= -this->dim() - 1 && dim < this->dim() + 1);

    if (dim < 0)
    {
        dim = dim + this->dim() + 1;
    }

    std::vector<int64_t> new_sizes = sizes();
    new_sizes.insert(std::next(new_sizes.begin(), dim), 1);

    int64_t stride_to_insert = (dim == 0) ? 1 : stride(dim - 1);

    std::vector<int64_t> new_strides = strides();
    new_strides.insert(std::next(new_strides.begin(), dim), stride_to_insert);

    std::shared_ptr<TensorImpl> new_impl = std::make_shared<TensorImpl>(
        impl_->storage_, impl_->storage_offset_, std::move(new_sizes), std::move(new_strides), options());

    return Tensor(new_impl);
}

Tensor Tensor::squeeze(int64_t dim) const
{
    CHECK_LT(dim, this->dim());
    CHECK_EQ(size(dim), 1);
    if (this->dim() == 1)
    {
        return *this;
    }

    std::vector<int64_t> new_sizes = sizes();
    new_sizes.erase(std::next(new_sizes.begin(), dim));

    std::vector<int64_t> new_strides = strides();
    new_strides.erase(std::next(new_strides.begin(), dim));

    std::shared_ptr<TensorImpl> new_impl = std::make_shared<TensorImpl>(
        impl_->storage_, impl_->storage_offset_, std::move(new_sizes), std::move(new_strides), options());

    return Tensor(new_impl);
}

Tensor Tensor::squeeze() const
{
    if (this->dim() == 1)
    {
        return *this;
    }

    std::vector<int64_t> new_sizes = sizes();
    new_sizes.erase(std::remove(new_sizes.begin(), new_sizes.end(), 1), new_sizes.end());

    std::vector<int64_t> new_strides = strides();
    new_strides.erase(std::remove(new_strides.begin(), new_strides.end(), 1), new_strides.end());

    std::shared_ptr<TensorImpl> new_impl = std::make_shared<TensorImpl>(
        impl_->storage_, impl_->storage_offset_, std::move(new_sizes), std::move(new_strides), options());

    return Tensor(new_impl);
}

bool Tensor::is_contiguous() const
{
    int64_t expected_stride = 1;
    for (int64_t i = dim() - 1; i >= 0; --i)
    {
        if (stride(i) != expected_stride)
        {
            return false;
        }
        expected_stride *= size(i);
    }
    return true;
}
Tensor Tensor::to(ScalarType new_type) const
{
    if (dtype() == new_type)
    {
        return *this;
    }
    return tinytorch::to(*this, new_type);
}
Tensor Tensor::sum() const
{
    return tinytorch::sum(*this);
}
Tensor Tensor::mean() const
{
    return tinytorch::mean(*this);
}
Tensor Tensor::std() const
{
    return tinytorch::std(*this);
}
void Tensor::fill_(double a)
{
    tinytorch::fill(*this, a);
}
Tensor Tensor::reshape(const SizeType& size) const
{
    SizeType new_sizes = size;
    fill_neg_one_dim(new_sizes, numel());

    Tensor result = empty(new_sizes, options());
    tinytorch::copy(*this, result);
    return result;
}

Tensor Tensor::repeat_interleave(int64_t count)
{
    return tinytorch::repeat_interleave(*this, count);
}
Tensor Tensor::transpose(int64_t dim0, int64_t dim1)
{
    return tinytorch::transpose(*this, dim0, dim1);
}
Tensor Tensor::clone() const
{
    Tensor result = empty_like(*this);
    tinytorch::copy(*this, result);
    return result;
}
void Tensor::copy_(Tensor a)
{
    tinytorch::copy(a, *this);
}
Tensor Tensor::min() const
{
    return tinytorch::min(*this);
}
Tensor Tensor::max() const
{
    return tinytorch::max(*this);
}
std::pair<Tensor, Tensor> Tensor::min(int64_t dim, bool keepdim) const
{
    return tinytorch::min(*this, dim, keepdim);
}
std::pair<Tensor, Tensor> Tensor::max(int64_t dim, bool keepdim) const 
{
    return tinytorch::max(*this, dim, keepdim);
}
Tensor Tensor::repeat(const SizeType& size) const
{
    return tinytorch::repeat(*this, size);
}
Tensor Tensor::index_add(int64_t dim, Tensor index, Tensor data) const
{
    return tinytorch::index_add(*this, dim, index, data);
}
void Tensor::backward() const
{
    tinytorch::backward(*this);
}
void Tensor::backward(Tensor t, bool retain_grad) const
{
    tinytorch::backward(*this, t);
}

}  // namespace tinytorch