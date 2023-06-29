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

namespace tinytorch
{

struct Edge;
struct Tensor;
struct TensorImpl;

struct SizeType
{
    SizeType() {}
    SizeType(std::vector<int64_t> v) : data_(v) {}
    int64_t& operator[](int64_t i) { return data_[i]; }
    const int64_t& operator[](int64_t i) const{ return data_[i]; }
    int64_t size() const { return data_.size(); }
    void resize(int64_t s) { data_.resize(s); }
    std::vector<int64_t> vec() const { return data_; }
    operator std::vector<int64_t>() const { return data_; }

   private:
    std::vector<int64_t> data_;
};
inline std::ostream& operator<<(std::ostream& strm, const SizeType& size)
{
    throw std::runtime_error("not implemented");
    return strm;
}



struct TINYTORCH_API Tensor
{
    // Tensor(int size = 0);
    // Tensor(std::vector<float> data);

    Tensor() {}
    Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    int64_t numel() const;
    // float& operator[](int idx);
    void resize(int new_size);


    const SizeType& sizes() const;

    ScalarType scalar_type() const;

        // void ClearGrad();
    const Tensor& grad() const;

    Tensor& mutable_grad();

    std::shared_ptr<Edge> getEdge();
    void SetEdge(std::shared_ptr<Edge> edge);

    template <typename T>
    T* data_ptr();

    uint8_t* ptr();

    int64_t dim() const;

    int64_t size(int64_t index) const;
    int64_t stride(int64_t index) const;

    void zero_();

    bool defined() const { return impl_ != nullptr; }

    TensorOptions options() const;

    void set_requires_grad(bool requires_grad);

    bool requires_grad() const ;
    int64_t element_size() const {
        throw std::runtime_error("not implemented");
        return 0;
    }

    Device device()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor view(std::vector<int64_t> sizes)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor slice(int64_t dim, int64_t start, int64_t end)
    {
        throw std::runtime_error("not implemented");
        return {};
    }

    Tensor clone()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor to(ScalarType new_type)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor to(Device new_type)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor permute(std::vector<int64_t> size)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor reshape(std::vector<int64_t> size)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor repeat(std::vector<int64_t> size)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor cpu()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor square()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor sqrt()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor cuda()
    {
        throw std::runtime_error("not implemented");
        return {};
    }

    Tensor clamp(double mi, double ma)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor clamp_min(double m)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor unsqueeze(int64_t dim)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor squeeze(int64_t dim)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor min()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    std::pair<Tensor, Tensor> min(int64_t index)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor max()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor sum()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor sum(int64_t dim, bool squeeze_dim)
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor mean()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor std()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    Tensor item()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    template <typename T>
    T item()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    double toDouble()
    {
        throw std::runtime_error("not implemented");
        return 0;
    }
    bool is_contiguous()
    {
        throw std::runtime_error("not implemented");
        return true;
    }
    bool is_cuda()
    {
        throw std::runtime_error("not implemented");
        return true;
    }
    bool is_cpu()
    {
        throw std::runtime_error("not implemented");
        return true;
    }

    Tensor contiguous()
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

    TensorImpl(SizeType sizes, TensorOptions options);
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
        return (T*)ptr();
    }

    uint8_t* ptr()
    {
        return (storage_->byte_ptr() + storage_offset_);
    }

    int64_t storage_offset_ = 0;
    std::shared_ptr<StorageImpl> storage_;
    SizeType sizes_;
    SizeType strides_;
    TensorOptions options_;
    // required for .backward()
    // std::vector<float> grad;
};

template <typename T>
T* Tensor::data_ptr()
{
    assert(impl_);
    assert(scalar_type() == kFloat);
    return (T*)ptr();
}

}  // namespace tinytorch
