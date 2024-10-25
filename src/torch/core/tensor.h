/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#define NOMINMAX

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
#include "torch/tiny_torch_config.h"

namespace tinytorch
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

namespace tinytorch
{

struct Edge;
struct Tensor;
struct TensorImpl;

struct CustomClassHolder
{
    TT_HD virtual ~CustomClassHolder() {}
};

struct TINYTORCH_API SizeType
{
    SizeType() {}
    SizeType(const std::vector<int64_t>& v) : data_(v) {}
    SizeType(std::vector<int64_t>&& v) : data_(std::move(v)) {}
    SizeType(const std::initializer_list<int64_t>& v) : data_(v) {}
    SizeType(const SizeType&) = default;
    SizeType(SizeType&&)      = default;

    int64_t& operator[](int64_t i)
    {
        if (i < 0)
        {
            i += size();
        }
        CHECK_GE(i, 0);
        CHECK_LT(i, size());
        return data_[i];
    }
    const int64_t& operator[](int64_t i) const
    {
        if (i < 0)
        {
            i += size();
        }
        CHECK_GE(i, 0);
        CHECK_LT(i, size());
        return data_[i];
    }
    int64_t size() const { return data_.size(); }
    void resize(int64_t s) { data_.resize(s); }
    std::vector<int64_t>& vec() { return data_; }
    const std::vector<int64_t>& vec() const { return data_; }
    operator const std::vector<int64_t>&() const { return data_; }
    int64_t numel() const
    {
        int64_t result = 1;
        for (auto v : data_)
        {
            result *= v;
        }
        return result;
    }

    SizeType& operator=(const SizeType&) = default;
    SizeType& operator=(SizeType&&)      = default;

   private:
    std::vector<int64_t> data_;
};

inline SizeType CompactStrideForSize(SizeType size)
{
    std::vector<int64_t> result(size.size());
    result.back() = 1;
    for (int64_t i = size.size() - 2; i >= 0; i--)
    {
        result[i] = result[i + 1] * size[i + 1];
    }
    return SizeType(result);
}

inline bool operator==(const SizeType& s1, const SizeType& s2)
{
    return s1.vec() == s2.vec();
}
inline std::ostream& operator<<(std::ostream& strm, const SizeType& size)
{
    strm << "[";
    for (int64_t i = 0; i < size.size(); ++i)
    {
        strm << size[i] << ((i < size.size() - 1) ? ", " : "");
    }
    strm << "]";
    return strm;
}


#undef min
#undef max


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

    Tensor grad() const;
    Tensor& mutable_grad();
    void set_grad(Tensor t);

    std::shared_ptr<Edge> getEdge() const;
    void SetEdge(std::shared_ptr<Edge> edge);

    template <typename T>
    T* data_ptr() const;

    void* data_ptr() const { return ptr(); }

    uint8_t* ptr() const;

    int64_t dim() const;

    int64_t size(int64_t index) const;
    int64_t stride(int64_t index) const;

    void zero_();

    bool defined() const { return impl_ != nullptr; }


    void retain_grad();
    void set_requires_grad(bool requires_grad);

    bool requires_grad() const;
    int64_t element_size() const;

    uint64_t AllocatorInfo();


    //  returns a view of the collapsed tensor and the new excluded dim
    std::pair<Tensor, int64_t> collapse_view(int64_t excludeDim = -1) const;
    Tensor view(const SizeType& sizes) const;
    Tensor operator[](int64_t index) const { return slice(0, index, index + 1).squeeze(0); }

    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    // no grad version
    Tensor slice_view(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor permute_view(const SizeType& size) const;

    // Tensor scatter_add(int64_t dim, Tensor ids, Tensor value) const
    // {
    //     throw std::runtime_error("not implemented");
    //     return {};
    // }
    Tensor& index_copy_(int64_t dim, Tensor ids, Tensor value);
    Tensor pow(Tensor a) const;
    void copy_(Tensor a, bool non_blocking = false);

    Tensor clone() const;

    Tensor to(ScalarType new_type, bool non_blocking = false) const;
    Tensor to(Device new_device, bool non_blocking = false) const;
    void to_(ScalarType new_type);
    void to_(Device new_device);

    Tensor flip(const SizeType& size) const;
    void resize_(const SizeType& size);
    Tensor permute(const SizeType& size) const;
    Tensor reshape(const SizeType& size) const;
    Tensor repeat(const SizeType& size) const;
    Tensor detach() const;

    Tensor& uniform_(double mi = 0, double ma = 1);

    Tensor cpu() const { return to(kCPU); }
    Tensor index_add(int64_t dim, Tensor index, Tensor data) const;
    Tensor square() const;
    Tensor sqrt() const;
    Tensor cuda() const { return to(kCUDA); }
    bool allclose(Tensor other, double rtol = 1e-4, double atol = 1e-07) const;


    Tensor round() const;

    Tensor clamp(double mi, double ma) const;
    void clamp_(double mi, double ma);
    Tensor clamp_min(double m) const;
    void clamp_min_(double m);
    Tensor clamp_max(double m) const;
    void clamp_max_(double m);

    Tensor gather(int64_t dim, Tensor index) const;
    Tensor norm(int64_t norm, int64_t dim, bool keepdim = false) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor squeeze(int64_t dim) const;
    Tensor squeeze() const;
    Tensor prod(int64_t dim, bool keepdim = false) const;
    Tensor cumprod(int64_t dim) const;
    Tensor cumsum(int64_t dim) const;
    void fill_(double a);

    Tensor min() const;
    Tensor max() const;


    std::pair<Tensor, Tensor> min(int64_t dim, bool keepdim = false) const;
    std::pair<Tensor, Tensor> max(int64_t dim, bool keepdim = false) const;
    Tensor std(int64_t index) const;

    Tensor sum() const;
    Tensor sum(int64_t dim, bool keepdim) const;
    Tensor sum(const SizeType& sizes, bool keepdim = false) const;
    Tensor mean() const;
    Tensor mean(int64_t dim, bool keepdim) const;
    Tensor mean(const SizeType& sizes, bool keepdim = false) const;
    Tensor std() const;
    Tensor index_select(int64_t i, Tensor index) const;
    Tensor abs() const;
    void set_data(Tensor t);
    Tensor repeat_interleave(int64_t count);
    Tensor transpose(int64_t dim0, int64_t dim1);
    void backward() const;
    void backward(Tensor t, bool retain_grad = false) const;
    Tensor item() const
    {
        CHECK_EQ(numel(), 1);
        return reshape({1});
    }
    template <typename T>
    T item() const
    {
        CHECK_EQ(numel(), 1);
        return *cpu().to(CppTypeToScalarType<T>::value).template data_ptr<T>();
    }
    double toDouble() const { return item<double>(); }
    double toFloat() const { return item<float>(); }
    int toInt() const { return item<int>(); }
    bool is_contiguous() const;
    bool is_leaf() const;
    inline bool is_cuda() const { return device().type() == kCUDA; } // Compare only type here, not device_index.
    bool is_cpu() const { return device() == kCPU; }

    Tensor contiguous() const;

   private:
    std::shared_ptr<TensorImpl> impl_;
};

template <typename T>
T* Tensor::data_ptr() const
{
    CHECK(impl_);

    auto dtype = scalar_type();
    (void)dtype;

    // TODO: Half!
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        CHECK_EQ(dtype, kUInt8);
    }
    else if constexpr (std::is_same_v<T, uint16_t>)
    {
        CHECK_EQ(dtype, kUInt16);
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
        CHECK_EQ(dtype, kInt16);
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        CHECK_EQ(dtype, kInt32);
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        CHECK_EQ(dtype, kLong);
    }
#ifdef __CUDACC__
    else if constexpr (std::is_same_v<T, half>)
    {
        CHECK_EQ(dtype, kHalf);
    }
#endif
    else if constexpr (std::is_same_v<T, Half>)
    {
        CHECK_EQ(dtype, kHalf);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        CHECK_EQ(dtype, kFloat);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        CHECK_EQ(dtype, kFloat64);
    }
    else
    {
        CHECK(false) << "invalid datatype " << typeid(T).name();
    }
    return (T*)ptr();
}

}  // namespace tinytorch
