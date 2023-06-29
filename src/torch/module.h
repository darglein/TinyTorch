/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "tensor.h"
#include <map>

namespace tinytorch
{
namespace nn
{

struct Module
{
    std::map<std::string, Tensor> named_parameters(){

        throw std::runtime_error("not implemented");
        return {};
    }

    std::vector<Tensor> parameters(){
        throw std::runtime_error("not implemented");
        return {};
    }
};

struct AnyModule
{

};

template <typename Contained>
class ModuleHolder{
   public:
    std::shared_ptr<Contained> impl_;

    ModuleHolder() {
        static_assert(
            std::is_default_constructible<Contained>::value,
            "You are trying to default construct a module which has "
            "no default constructor. Use = nullptr to give it the empty state "
            "(e.g. `Linear linear = nullptr;` instead of `Linear linear;`).");
    }

    template <
        typename Head,
        typename... Tail>
    explicit ModuleHolder(Head&& head, Tail&&... tail)
        : impl_(new Contained(
              std::forward<Head>(head),
              std::forward<Tail>(tail)...)) {}

    /// Constructs the `ModuleHolder` from a pointer to the contained type.
    /// Example: `Linear(std::make_shared<LinearImpl>(...))`.
    /* implicit */ ModuleHolder(std::shared_ptr<Contained> module)
        : impl_(std::move(module)) {}

    /// Constructs the `ModuleHolder` with an empty contained value. Access to
    /// the underlying module is not permitted and will throw an exception, until
    /// a value is assigned.
    /* implicit */ ModuleHolder(std::nullptr_t) : impl_(nullptr) {}

    /// Returns true if the `ModuleHolder` contains a module, or false if it is
    /// `nullptr`.
    explicit operator bool() const noexcept {
        return !is_empty();
    }

    /// Forwards to the contained module.
    Contained* operator->() {
        return get();
    }

    /// Forwards to the contained module.
    const Contained* operator->() const {
        return get();
    }

    /// Returns a reference to the contained module.
    Contained& operator*() {
        return *get();
    }

    /// Returns a const reference to the contained module.
    const Contained& operator*() const {
        return *get();
    }
    /// Returns a pointer to the underlying module.
    Contained* get() {
        return impl_.get();
    }

    /// Returns a const pointer to the underlying module.
    const Contained* get() const {
        return impl_.get();
    }
    /// Returns true if the `ModuleHolder` does not contain a module.
    bool is_empty() const noexcept {
        return impl_ == nullptr;
    }
};

/// Defines a class `Name` which inherits from `nn::ModuleHolder` to provide a
/// wrapper over a `std::shared_ptr<ImplType>`.
/// `Impl` is a type alias for `ImplType` which provides a way to call static
/// method of `ImplType`.
#define TORCH_MODULE_IMPL(Name, ImplType)                              \
  class Name : public torch::nn::ModuleHolder<ImplType> { /* NOLINT */ \
   public:                                                             \
    using torch::nn::ModuleHolder<ImplType>::ModuleHolder;             \
  }


/// Like `TORCH_MODULE_IMPL`, but defaults the `ImplType` name to `<Name>Impl`.
#define TORCH_MODULE(Name) TORCH_MODULE_IMPL(Name, Name##Impl)


}  // namespace nn
}  // namespace tinytorch
