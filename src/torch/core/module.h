/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "torch/core/tensor.h"

#include <map>

namespace tinytorch
{

inline void set_num_threads(int n)
{
    // std::cerr << "tinytorch::set_num_threads is currently ignored!" << std::endl;
}

namespace nn
{

template <bool value, typename T = void>
using disable_if_t = std::enable_if_t<!value, T>;
struct ModuleHolderIndicator
{
};
// A type trait that is true for types that are `ModuleHolder`s.
template <typename T>
using is_module_holder = std::is_base_of<ModuleHolderIndicator, std::decay_t<T>>;

template <typename T>
using disable_if_module_holder_t = disable_if_t<is_module_holder<T>::value>;

// Base template.
template <bool is_module_holder_value, typename T, typename C>
struct is_module_holder_of_impl;

// False branch. `T` is not a `ModuleHolder` and thus not a `ModuleHolder` with
// contained type `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<false, T, C> : std::false_type
{
};

// True branch. `T` is a `ModuleHolder` and thus we can legit access its
// `ContainedType` and compare it against `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<true, T, C> : std::is_same<typename T::ContainedType, C>
{
};

// Helper template.
template <typename T, typename C>
struct is_module_holder_of : is_module_holder_of_impl<is_module_holder<T>::value, std::decay_t<T>, std::decay_t<C>>
{
};

template <typename Contained>
class ModuleHolder : ModuleHolderIndicator
{
   public:
    using ContainedType = Contained;
    std::shared_ptr<Contained> impl_;

    std::shared_ptr<Contained> ptr() { return impl_; }

    ModuleHolder()
    {
        static_assert(std::is_default_constructible<Contained>::value,
                      "You are trying to default construct a module which has "
                      "no default constructor. Use = nullptr to give it the empty state "
                      "(e.g. `Linear linear = nullptr;` instead of `Linear linear;`).");
    }

    template <typename Head, typename... Tail,
              typename = typename std::enable_if<!(is_module_holder_of<Head, ContainedType>::value &&
                                                   (sizeof...(Tail) == 0))>::type>
    explicit ModuleHolder(Head&& head, Tail&&... tail)
        : impl_(new Contained(std::forward<Head>(head), std::forward<Tail>(tail)...))
    {
    }

    /// Constructs the `ModuleHolder` from a pointer to the contained type.
    /// Example: `Linear(std::make_shared<LinearImpl>(...))`.
    /* implicit */ ModuleHolder(std::shared_ptr<Contained> module) : impl_(std::move(module)) {}

    /// Constructs the `ModuleHolder` with an empty contained value. Access to
    /// the underlying module is not permitted and will throw an exception, until
    /// a value is assigned.
    /* implicit */ ModuleHolder(std::nullptr_t) : impl_(nullptr) {}

    /// Returns true if the `ModuleHolder` contains a module, or false if it is
    /// `nullptr`.
    explicit operator bool() const noexcept { return !is_empty(); }

    /// Forwards to the contained module.
    Contained* operator->() { return get(); }

    /// Forwards to the contained module.
    const Contained* operator->() const { return get(); }

    /// Returns a reference to the contained module.
    Contained& operator*() { return *get(); }

    /// Returns a const reference to the contained module.
    const Contained& operator*() const { return *get(); }
    /// Returns a pointer to the underlying module.
    Contained* get() { return impl_.get(); }

    /// Returns a const pointer to the underlying module.
    const Contained* get() const { return impl_.get(); }
    /// Returns true if the `ModuleHolder` does not contain a module.
    bool is_empty() const noexcept { return impl_ == nullptr; }
};

struct Module
{
    virtual ~Module() {}
    std::map<std::string, Tensor> named_parameters()
    {
        throw std::runtime_error("not implemented");
        return {};
    }

    void to(Device d)
    {
        for (auto& b : buffers_)
        {
            b.second.to_(d);
        }
        for (auto& b : parameters_)
        {
            b.second.to_(d);
        }
        for (auto& b : modules_)
        {
            b.second->to(d);
        }
    }
    void zero_grad(bool set_to_none = false)
    {
        for (auto& b : parameters_)
        {
            if (b.second.grad().defined())
            {
                if (set_to_none)
                {
                    b.second.mutable_grad() = Tensor();
                }
                else
                {
                    b.second.mutable_grad().zero_();
                }
            }
        }
        for (auto& b : modules_)
        {
            b.second->zero_grad(set_to_none);
        }
    }
    void train(bool on = true)
    {
        is_training_ = true;
        for (auto& b : modules_)
        {
            b.second->train(on);
        }
    }

    bool is_training() const { return is_training_; }

    void register_buffer(std::string name, Tensor& t) { buffers_[name] = t; }
    void register_parameter(std::string name, Tensor& t)
    {
        t.set_requires_grad(true);
        parameters_[name] = t;
    }

    template <typename ModuleType>
    std::shared_ptr<ModuleType> register_module(std::string name, std::shared_ptr<ModuleType> module)
    {
        modules_[name] = module;
        return module;
    }

    template <typename ModuleType>
    std::shared_ptr<ModuleType> register_module(std::string name, ModuleHolder<ModuleType> module_holder)
    {
        return register_module(std::move(name), module_holder.ptr());
    }

    template <typename ModuleType>
    std::shared_ptr<ModuleType> replace_module(std::string name, ModuleHolder<ModuleType> module_holder)
    {
        modules_[name] = module_holder.ptr();
        return module_holder.ptr();
    }
    std::vector<Tensor> parameters()
    {
        std::vector<Tensor> result;
        for (auto b : parameters_)
        {
            result.push_back(b.second);
        }
        return result;
    }

    std::vector<Tensor> buffers()
    {
        std::vector<Tensor> result;
        for (auto b : buffers_)
        {
            result.push_back(b.second);
        }
        return result;
    }

    std::vector<std::shared_ptr<Module>> children()
    {
        std::vector<std::shared_ptr<Module>> result;
        for (auto b : modules_)
        {
            result.push_back(b.second);
        }
        return result;
    }

    std::string name() { return typeid(*this).name(); }

    std::map<std::string, Tensor> buffers_;
    std::map<std::string, Tensor> parameters_;
    std::map<std::string, std::shared_ptr<Module>> modules_;
    bool is_training_ = true;
};

struct AnyModule
{
};


/// Defines a class `Name` which inherits from `nn::ModuleHolder` to provide a
/// wrapper over a `std::shared_ptr<ImplType>`.
/// `Impl` is a type alias for `ImplType` which provides a way to call static
/// method of `ImplType`.
#define TORCH_MODULE_IMPL(Name, ImplType)                      \
    class Name : public torch::nn::ModuleHolder<ImplType>      \
    { /* NOLINT */                                             \
       public:                                                 \
        using torch::nn::ModuleHolder<ImplType>::ModuleHolder; \
    }


/// Like `TORCH_MODULE_IMPL`, but defaults the `ImplType` name to `<Name>Impl`.
#define TORCH_MODULE(Name) TORCH_MODULE_IMPL(Name, Name##Impl)


}  // namespace nn



template <typename ModuleType>
inline void load(nn::ModuleHolder<ModuleType> module_holder, std::string name)
{
    throw std::runtime_error("not implemented");
}

template <typename ModuleType>
inline void save(nn::ModuleHolder<ModuleType> module_holder, std::string name)
{
    throw std::runtime_error("not implemented");
}

inline void save(Tensor t, std::string name)
{
    throw std::runtime_error("not implemented");
}


}  // namespace tinytorch
