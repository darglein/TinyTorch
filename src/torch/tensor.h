/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "assert.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tensor_data.h"
#include "tensor_options.h"
#include "tiny_torch_config.h"

namespace TINY_TORCH_NAMESPACE
{

struct TensorImpl;
}
namespace std
{
// This removes a warning on MSVC:
// warning C4251: 'tinytorch::Tensor::impl_': class 'std::shared_ptr<tinytorch::TensorImpl>'
// needs to have dll-interface to be used by clients of struct 'tinytorch::Tensor'
template class TINYTORCH_API shared_ptr<tinytorch::TensorImpl>;
}  // namespace std

namespace TINY_TORCH_NAMESPACE
{

struct Edge;
struct Tensor;
struct TensorImpl;

struct CustomClassHolder
{
};

struct SizeType
{
    SizeType() {}
    SizeType(const std::vector<int64_t>& v) : data_(v) {}
    SizeType(const std::initializer_list<int64_t>& v) : data_(v) {}
    SizeType(const SizeType&) = default;
    SizeType(SizeType&&)      = default;
    int64_t& operator[](int64_t i) { return data_[i]; }
    const int64_t& operator[](int64_t i) const { return data_[i]; }
    int64_t size() const { return data_.size(); }
    void resize(int64_t s) { data_.resize(s); }
    const std::vector<int64_t>& vec() const { return data_; }
    operator const std::vector<int64_t>&() const { return data_; }

    SizeType& operator=(const SizeType&) = default;
    SizeType& operator=(SizeType&&)      = default;

   private:
    std::vector<int64_t> data_;
};
inline bool operator==(SizeType s1, SizeType s2)
{
    return s1.vec() == s2.vec();
}
inline std::ostream& operator<<(std::ostream& strm, const SizeType& size)
{
    strm << "[ ";
    for (int64_t i = 0; i < size.size(); ++i)
    {
        strm << size[i] << ((i < size.size() - 1) ? ", " : " ");
    }
    strm << "]";
    return strm;
}



struct TINYTORCH_API Tensor
{
    Tensor() {}
    Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    int64_t numel() const;

    const SizeType& strides() const;
    const SizeType& sizes() const;

    ScalarType scalar_type() const;
    ScalarType dtype() const { return scalar_type(); }

    Device device() const;
    TensorOptions options() const;

    const Tensor& grad() const;

    Tensor& mutable_grad();

    std::shared_ptr<Edge> getEdge() const;
    void SetEdge(std::shared_ptr<Edge> edge);

    template <typename T>
    T* data_ptr() const;

    uint8_t* ptr() const;

    int64_t dim() const;

    int64_t size(int64_t index) const;
    int64_t stride(int64_t index) const;

    void zero_();

    bool defined() const { return impl_ != nullptr; }

    void set_requires_grad(bool requires_grad);

    bool requires_grad() const;
    int64_t element_size() const;

    Tensor view(const SizeType& sizes) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor operator[](int64_t dim) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor select(int64_t dim, int64_t index) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor scatter_add(int64_t dim, Tensor ids, Tensor value) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    void index_copy_(int64_t dim, Tensor ids, Tensor value) { throw std::runtime_error("not implemented"); }

    void copy_(Tensor a) { throw std::runtime_error("not implemented"); }

    Tensor clone() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor to(ScalarType new_type) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor to(Device new_type) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor flip(const SizeType& size) const { throw std::runtime_error("not implemented"); }
    void resize_(const SizeType& size) { throw std::runtime_error("not implemented"); }
    Tensor permute(const SizeType& size) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor reshape(const SizeType& size) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor repeat(const SizeType& size) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor detach() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor& uniform_(double mi, double ma)
    {
        throw std::runtime_error("not implemented");
        return *this;
    }
    Tensor cpu() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor index_add(int64_t dim, Tensor index, Tensor data) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor square() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor sqrt() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor cuda() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }

    Tensor clamp(double mi, double ma) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor clamp_min(double m) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor clamp_max_(double m) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor norm(int64_t norm, int64_t dim, bool keep) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor unsqueeze(int64_t dim) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor squeeze(int64_t dim) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor cumprod(int64_t dim) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor cumsum(int64_t dim) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor fill_(double a) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor min() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    std::pair<Tensor, Tensor> min(int64_t index) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor max() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor sum() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor sum(int64_t dim, bool squeeze_dim) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor mean() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor mean(int64_t dim, bool squeeze_dim) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor mean(SizeType sizes) const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor abs() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor set_data(Tensor t)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    void backward() const { throw std::runtime_error("not implemented"); }
    void backward(Tensor t) const { throw std::runtime_error("not implemented"); }
    Tensor std() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor item() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    template <typename T>
    T item() const
    {
        assert(numel() == 1);
        return *data_ptr<T>();
    }
    double toDouble() const
    {
        throw std::runtime_error("not implemented");
        return 0;
    }
    double toFloat() const
    {
        throw std::runtime_error("not implemented");
        return 0;
    }
    int toInt() const
    {
        throw std::runtime_error("not implemented");
        return 0;
    }
    bool is_contiguous() const
    {
        throw std::runtime_error("not implemented");
        return true;
    }
    inline bool is_cuda() const { return device() == kCUDA; }
    bool is_cpu() const { return device() == kCPU; }

    Tensor contiguous() const
    {
        throw std::runtime_error("not implemented");
        return {};
    }

   private:
    std::shared_ptr<TensorImpl> impl_;
};


struct AutogradMeta
{
    Tensor _grad;
    std::shared_ptr<Edge> edge;
    bool _requires_grad = false;

    Tensor& mutable_grad() { return _grad; }
    const Tensor& grad() const { return _grad; }
};

struct TensorImpl
{
    TensorImpl(const SizeType& sizes, TensorOptions options);
    // TensorImpl(std::vector<float> data) : data(data) {}


    void set_requires_grad(bool requires_grad);

    bool requires_grad() const
    {
        if (autograd_meta)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    int64_t dim() const { return sizes_.size(); }


    std::unique_ptr<AutogradMeta> autograd_meta;

    int64_t numel() const
    {
        int64_t res = 1;
        for (auto v : sizes_.vec()) res *= v;
        return res;
    }


    template <typename T>
    T* data_ptr()
    {
        return (T*)data_ptr();
    }


    uint8_t* data_ptr() { return (storage_->byte_ptr() + storage_offset_); }

    int64_t storage_offset_ = 0;
    std::shared_ptr<StorageImpl> storage_;
    SizeType sizes_;
    SizeType strides_;
    TensorOptions options_;
    // required for .backward()
    // std::vector<float> grad;
};

template <typename T>
T* Tensor::data_ptr() const
{
    assert(impl_);

    auto dtype = scalar_type();

    // TODO: Half!
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        assert(dtype == kUInt8);
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
        assert(dtype == kInt16);
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        assert(dtype == kInt32);
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        assert(dtype == kLong);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        assert(dtype == kFloat);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        assert(dtype == kFloat64);
    }
    else
    {
        static_assert(false);
    }
    return (T*)ptr();
}

}  // namespace TINY_TORCH_NAMESPACE
