/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tensor.h"

#include "torch/core/backward.h"
#include "torch/core/graph.h"
#include "torch/core/ops/all.h"
#include "torch/core/tensor.h"

#include "../tiny_torch_cuda.h"
#include "torch/core/tensor_impl.h"

namespace tinytorch
{
int64_t Tensor::numel() const
{
    CHECK(defined());
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
    CHECK(impl_->autograd_meta->mutable_grad().defined());
    return impl_->autograd_meta->mutable_grad();
}


void Tensor::set_grad(Tensor t)
{
    CHECK(impl_->autograd_meta);
    CHECK(!impl_->autograd_meta->_grad.defined());
    impl_->autograd_meta->_grad = t;
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
    CHECK(impl_);
    impl_->set_requires_grad(requires_grad);
}
bool Tensor::requires_grad() const
{
    CHECK(impl_);
    return impl_->requires_grad();
}
int64_t Tensor::element_size() const
{
    return elementSize(dtype());
}
Device Tensor::device() const
{
    CHECK(impl_);
    return impl_->options_.device_;
}
uint8_t* Tensor::ptr() const
{
    CHECK(impl_);
    return impl_->data_ptr();
}
const SizeType& Tensor::strides() const
{
    CHECK(impl_);
    return impl_->strides_;
}
const SizeType& Tensor::sizes() const
{
    CHECK(impl_);
    return impl_->sizes_;
}
int64_t Tensor::dim() const
{
    CHECK(impl_);
    return impl_->dim();
}
int64_t Tensor::size(int64_t index) const
{
    CHECK(impl_);
    return impl_->sizes_[index];
}
int64_t Tensor::stride(int64_t index) const
{
    CHECK(impl_);
    return impl_->strides_[index];
}
void Tensor::zero_()
{
    fill(*this, 0);
}
ScalarType Tensor::scalar_type() const
{
    CHECK(impl_);
    return impl_->options_.dtype_;
}
TensorOptions Tensor::options() const
{
    CHECK(impl_);
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


std::pair<Tensor, int> Tensor::collapse_view(int excludeDim) const
{
    auto sizes   = this->sizes();
    auto strides = this->strides();

    int64_t stopDim             = (excludeDim == -1) ? dim() : excludeDim;
    int64_t newIndex            = -1;
    int64_t oldIndex            = 0;
    int64_t remappedExcludedDim = -1;

    while (oldIndex < dim())
    {
        // Finds a dimension to collapse into
        for (; oldIndex < stopDim; ++oldIndex)
        {
            if (sizes[oldIndex] == 1)
            {
                continue;
            }

            ++newIndex;
            sizes[newIndex]   = sizes[oldIndex];
            strides[newIndex] = strides[oldIndex];
            ++oldIndex;
            break;
        }

        // Collapses dims
        for (; oldIndex < stopDim; ++oldIndex)
        {
            if (sizes[oldIndex] == 1)
            {
                continue;
            }

            if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex])
            {
                sizes[newIndex] *= sizes[oldIndex];
                strides[newIndex] = strides[oldIndex];
            }
            else
            {
                ++newIndex;
                sizes[newIndex]   = sizes[oldIndex];
                strides[newIndex] = strides[oldIndex];
            }
        }

        // Handles excludeDim being set (oldIndex == excludeDim)
        if (oldIndex != dim())
        {
            // Preserves excluded dimension
            ++newIndex;
            sizes[newIndex]     = sizes[oldIndex];
            strides[newIndex]   = strides[oldIndex];
            remappedExcludedDim = newIndex;

            // Restarts iteration after excludeDim
            ++oldIndex;
            stopDim = dim();
        }
    }

    // Handles special case of all dims size 1
    if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1))
    {
        remappedExcludedDim = 0;
        sizes[0]            = 1;
        strides[0]          = 1;
        sizes.resize(1);
        strides.resize(1);
    }
    else
    {
        sizes.resize(newIndex + 1);
        strides.resize(newIndex + 1);
    }

    std::shared_ptr<TensorImpl> new_impl =
        TensorImpl::create(impl_->storage_, impl_->storage_offset_, std::move(sizes), std::move(strides), options());

    return {Tensor(new_impl), remappedExcludedDim};
}

Tensor Tensor::view(const SizeType& sizes) const
{
    CHECK(!this->requires_grad() || !GradMode::is_enabled());
    SizeType new_sizes = sizes;
    fill_neg_one_dim(new_sizes, numel());

    CHECK(is_contiguous()) << "Invalid view() call. Use Reshape() instead!";

    SizeType new_strides;
    new_strides.resize(new_sizes.vec().size());
    int64_t stride = 1;
    for (int64_t i = new_strides.vec().size() - 1; i >= 0; --i)
    {
        new_strides[i] = stride;
        stride *= new_sizes[i];
    }

    std::shared_ptr<TensorImpl> new_impl = TensorImpl::create(impl_->storage_, impl_->storage_offset_,
                                                              std::move(new_sizes), std::move(new_strides), options());

    return Tensor(new_impl);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const
{
    return tinytorch::slice(*this, dim, start, end, step);
}

Tensor Tensor::slice_view(int64_t dim, int64_t start, int64_t end, int64_t step) const
{
    CHECK(!this->requires_grad() || !GradMode::is_enabled());
    int64_t dims = this->dim();

    if (dim < 0)
    {
        dim += this->dim();
    }

    CHECK_GE(dim, 0);
    CHECK_LT(dim, dims);
    CHECK_LE(start, end);
    CHECK_LE(end, this->size(dim));
    CHECK_EQ((end - start) % step, 0);

    int64_t offset = start * this->stride(dim);
    offset *= this->element_size();

    std::vector<int64_t> new_sizes = this->sizes();
    new_sizes[dim]                 = (end - start) / step;

    auto new_strides = this->strides();
    new_strides[dim] *= step;

    std::shared_ptr<TensorImpl> new_impl = TensorImpl::create(impl_->storage_, impl_->storage_offset_ + offset,
                                                              std::move(new_sizes), std::move(new_strides), options());

    return Tensor(new_impl);
}

Tensor Tensor::permute_view(const SizeType& index) const
{
    CHECK_EQ(dim(), index.size());
    auto new_sizes   = this->sizes();
    auto new_strides = this->strides();

    for (int i = 0; i < dim(); ++i)
    {
        new_sizes[i]   = size(index[i]);
        new_strides[i] = stride(index[i]);
    }

    std::shared_ptr<TensorImpl> new_impl = TensorImpl::create(impl_->storage_, impl_->storage_offset_,
                                                              std::move(new_sizes), std::move(new_strides), options());
    return new_impl;
}

void Tensor::resize_(const SizeType& size)
{
    this->impl_->autograd_meta = nullptr;
    auto new_tensor = empty(size, options());

    int64_t elements_to_copy = std::min(this->numel(), new_tensor.numel());

    new_tensor.view({-1}).slice(0, 0, elements_to_copy).copy_(this->view({-1}).slice(0, 0, elements_to_copy));



    this->set_data(new_tensor);
    // std::shared_ptr<TensorImpl> new_impl = TensorImpl::create(size, options());
    // impl_->set_data(*new_impl);
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
    return reshape(new_sizes);
}

Tensor Tensor::squeeze(int64_t dim) const
{
    if (dim < 0)
    {
        dim += this->dim();
    }

    CHECK_LT(dim, this->dim());
    CHECK_GE(dim, 0);
    CHECK_EQ(size(dim), 1);
    if (this->dim() == 1)
    {
        return *this;
    }

    std::vector<int64_t> new_sizes = sizes();
    new_sizes.erase(std::next(new_sizes.begin(), dim));

    return reshape(new_sizes);
}

Tensor Tensor::squeeze() const
{
    if (this->dim() == 1)
    {
        return *this;
    }

    std::vector<int64_t> new_sizes = sizes();
    new_sizes.erase(std::remove(new_sizes.begin(), new_sizes.end(), 1), new_sizes.end());

    return reshape(new_sizes);
}

bool Tensor::is_contiguous() const
{
    int64_t expected_stride = 1;
    for (int64_t i = dim() - 1; i >= 0; --i)
    {
        if (size(i) != 1 && stride(i) != expected_stride)
        {
            return false;
        }
        expected_stride *= size(i);
    }
    return true;
}
Tensor Tensor::contiguous() const
{
    if (is_contiguous())
    {
        return *this;
    }

    return clone();
}
Tensor Tensor::to(ScalarType new_type, bool non_blocking) const
{
    if (dtype() == new_type)
    {
        return *this;
    }
    return tinytorch::to(*this, new_type, non_blocking);
}
Tensor Tensor::to(Device new_device, bool non_blocking) const
{
    if (!impl_)
    {
        return *this;
    }

    return tinytorch::to(*this, new_device, non_blocking);
}

void Tensor::to_(ScalarType new_type)
{
    auto result = to(new_type);
    result.set_requires_grad(requires_grad());
    impl_->set_data(*result.impl_);
}
void Tensor::to_(Device new_device)
{
    auto result = to(new_device);
    result.set_requires_grad(requires_grad());
    impl_->set_data(*result.impl_);
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
Tensor Tensor::abs() const
{
    return tinytorch::abs(*this);
}
void Tensor::fill_(double a)
{
    tinytorch::fill(*this, a);
}
Tensor Tensor::reshape(const SizeType& size) const
{
    return tinytorch::reshape(*this, size);
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
    return tinytorch::clone(*this);
}
void Tensor::copy_(Tensor a, bool non_blocking)
{
    tinytorch::copy(a, *this, non_blocking);
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
bool Tensor::allclose(Tensor other, double rtol, double atol) const
{
    Tensor diff = (*this - other).abs().max();
    double v    = 0.f;
    switch (diff.dtype())
    {
        case kFloat:
            v = diff.toFloat();
            break;
        case kDouble:
            v = diff.toDouble();
            break;
        default:
            CHECK(false);
    }
    return v < atol;
}
void Tensor::backward() const
{
    tinytorch::backward(*this);
}
void Tensor::backward(Tensor t, bool retain_grad) const
{
    tinytorch::backward(*this, t, retain_grad);
}
Tensor Tensor::std(int64_t index) const
{
    return tinytorch::std(*this, index);
}
Tensor Tensor::sum(int64_t dim, bool keepdim) const
{
    return tinytorch::sum(*this, dim, keepdim);
}
Tensor Tensor::sum(const SizeType& sizes, bool keepdim) const
{
    return tinytorch::sum(*this, sizes, keepdim);
}
Tensor Tensor::mean(int64_t dim, bool keepdim) const
{
    return tinytorch::mean(*this, dim, keepdim);
}
Tensor Tensor::mean(const SizeType& sizes, bool keepdim) const
{
    return tinytorch::mean(*this, sizes, keepdim);
}
Tensor Tensor::index_select(int64_t i, Tensor index) const
{
    return tinytorch::index_select(*this, i, index);
}
Tensor Tensor::pow(Tensor a) const
{
    return tinytorch::pow(*this, a);
}
Tensor Tensor::permute(const SizeType& size) const
{
    return tinytorch::permute(*this, size);
}
Tensor Tensor::detach() const
{
    auto result = clone();
    result.set_requires_grad(false);
    return result;
}
Tensor& Tensor::uniform_(double mi, double ma)
{
    tinytorch::uniform(*this, mi, ma);
    return *this;
}
Tensor Tensor::square() const
{
    return tinytorch::square(*this);
}
Tensor Tensor::sqrt() const
{
    return tinytorch::sqrt(*this);
}
Tensor Tensor::clamp(double mi, double ma) const
{
    return tinytorch::clamp(*this, mi, ma);
}
void Tensor::clamp_(double mi, double ma)
{
    tinytorch::clamp_(*this, mi, ma);
}
Tensor Tensor::clamp_min(double m) const
{
    return clamp(m, std::numeric_limits<double>::infinity());
}
void Tensor::clamp_min_(double m)
{
    clamp_(m, std::numeric_limits<double>::infinity());
}
Tensor Tensor::clamp_max(double m) const
{
    return clamp(-std::numeric_limits<double>::infinity(), m);
}
void Tensor::clamp_max_(double m)
{
    clamp_(-std::numeric_limits<double>::infinity(), m);
}
Tensor Tensor::norm(int64_t norm, int64_t dim, bool keepdim) const
{
    return tinytorch::norm(*this, norm, dim, keepdim);
}
Tensor Tensor::cumprod(int64_t dim) const
{
    return tinytorch::cumprod(*this, dim);
}
Tensor Tensor::cumsum(int64_t dim) const
{
    return tinytorch::cumsum(*this, dim);
}
bool Tensor::is_leaf() const
{
    if (!requires_grad()) return false;

    if (!impl_->autograd_meta->edge) return false;

    std::shared_ptr<autograd::AccumulateGrad> node =
        std::dynamic_pointer_cast<autograd::AccumulateGrad>(impl_->autograd_meta->edge->function);

    return node != nullptr;
}
void Tensor::set_data(Tensor t)
{
    this->impl_->set_data(*t.impl_);
}
Tensor Tensor::round() const
{
    return tinytorch::round(*this);
}
Tensor& Tensor::index_copy_(int64_t dim, Tensor ids, Tensor value)
{
    tinytorch::index_copy(*this, dim, ids, value);
    return *this;
}
void Tensor::retain_grad()
{
    CHECK(requires_grad());
    CHECK(!is_leaf());
    CHECK(impl_->autograd_meta->edge);
    impl_->autograd_meta->_retain_grad = true;

    auto old_edge = impl_->autograd_meta->edge;

    auto accu_node = std::make_shared<autograd::AccumulateGrad>(impl_);
    accu_node->next.push_back(old_edge);
    accu_node->num_inputs_of_forward = 1;

    auto intermedieate_edge    = std::make_shared<Edge>(accu_node, 0);
    impl_->autograd_meta->edge = intermedieate_edge;
}

Tensor Tensor::gather(int64_t dim, Tensor index) const
{
    return tinytorch::gather(*this, dim, index);
}
Tensor Tensor::prod(int64_t dim, bool keepdim) const
{
    return tinytorch::prod(*this, dim, keepdim);
}


}  // namespace tinytorch