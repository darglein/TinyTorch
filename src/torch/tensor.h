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
#include "tiny_torch_config.h"
#include "tensor_options.h"

namespace tinytorch
{

struct Edge;
struct Tensor;
struct TensorImpl;


struct TINYTORCH_API Tensor
{
    // Tensor(int size = 0);
    // Tensor(std::vector<float> data);

    Tensor() {}
    Tensor(std::shared_ptr<TensorImpl> impl) : impl_(impl) {}
    int64_t numel() const;
    // float& operator[](int idx);
    void resize(int new_size);


    const std::vector<int64_t>& sizes() const;

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
    bool requires_grad() const;

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
    TensorImpl(const std::vector<int64_t>& sizes, TensorOptions options);
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
        for (auto v : sizes_) res *= v;
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
    std::vector<int64_t> sizes_;
    std::vector<int64_t> strides_;
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
