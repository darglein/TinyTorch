/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#define NOMINMAX

#include "tensor.h"

namespace tinytorch
{


struct AutogradMeta
{
    Tensor _grad;
    std::shared_ptr<Edge> edge;
    bool _requires_grad = false;
    bool _retain_grad   = false;

    Tensor& mutable_grad() { return _grad; }
    const Tensor& grad() const { return _grad; }
};

struct TensorImpl : public std::enable_shared_from_this<TensorImpl>
{
   private:
    TensorImpl(const SizeType& sizes, TensorOptions options);
    TensorImpl(std::shared_ptr<StorageImpl> storage, int64_t storage_offset, const SizeType& sizes,
               const SizeType& strides, TensorOptions options);

   public:
    template <typename... Args>
    [[nodiscard]] static std::shared_ptr<TensorImpl> create(Args&&... args)
    {
        // Not using std::make_shared<Best> because the c'tor is private.
        return std::shared_ptr<TensorImpl>(new TensorImpl(std::forward<Args>(args)...));
    }
    std::shared_ptr<TensorImpl> getptr() { return shared_from_this(); }


    void set_requires_grad(bool requires_grad);
    bool requires_grad() const;

    int64_t dim() const { return sizes_.size(); }

    int64_t numel() const { return sizes_.numel(); }


    // overwrite the current data (storage, sizes,...)
    // Note: this changes the data for all "Tensor" objects that point to this impl
    void set_data(TensorImpl& other);



    template <typename T>
    T* data_ptr()
    {
        return (T*)data_ptr();
    }
    uint8_t* data_ptr() { return (storage_->byte_ptr() + storage_offset_); }

    // in byte
    int64_t storage_offset_ = 0;
    std::shared_ptr<StorageImpl> storage_;
    SizeType sizes_;
    SizeType strides_;
    TensorOptions options_;

    std::unique_ptr<AutogradMeta> autograd_meta;


   private:
    void recompute_strides();
};

}  // namespace tinytorch
