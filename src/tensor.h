#pragma once

#include "assert.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>


namespace tinytorch
{

struct Edge;

struct TensorImpl
{
    TensorImpl(int size) : data(size) {}
    TensorImpl(std::vector<float> data) : data(data) {}

    std::vector<float> data;

    // required for .backward()
    std::vector<float> grad;
    std::shared_ptr<Edge> edge;
};

struct Tensor
{
    Tensor(int size = 0) : impl_(std::make_shared<TensorImpl>(size)) {}
    Tensor(std::vector<float> data) : impl_(std::make_shared<TensorImpl>(data)) {}
    int size() { return impl_->data.size(); }
    float& operator[](int idx) { return impl_->data[idx]; }
    void resize(int new_size)
    {
        assert(impl_);
        impl_->data.resize(new_size, 0);
        impl_->grad.resize(new_size, 0);
    }


    void ClearGrad() { impl_->grad.clear(); }
    Tensor grad() { return Tensor(impl_->grad); }

    void AddGradInplace(Tensor g)
    {
        resize(g.size());
        for (int i = 0; i < size(); ++i)
        {
            impl_->grad[i] += g[i];
        }
    }
    void AddInplace(Tensor g)
    {
        resize(g.size());
        for (int i = 0; i < size(); ++i)
        {
            impl_->data[i] += g[i];
        }
    }
    std::shared_ptr<Edge> getEdge() { return impl_->edge; };
    void SetEdge(std::shared_ptr<Edge> edge) { impl_->edge = edge; }

   private:
    std::shared_ptr<TensorImpl> impl_;
};

// ================================================================================
// Tensor Create operators

inline Tensor zero(int size)
{
    Tensor t(size);
    for (int i = 0; i < t.size(); ++i)
    {
        t[i] = 0;
    }
    return t;
}


inline Tensor rand(int size)
{
    Tensor t(size);


    static std::mt19937 mersenne_engine{572547235};
    std::uniform_real_distribution<float> dist{0.f, 1.f};

    for (int i = 0; i < t.size(); ++i)
    {
        t[i] = dist(mersenne_engine);
    }

    return t;
}

inline std::ostream& operator<<(std::ostream& strm, Tensor t)
{
    strm << "[Tensor s=" << t.size() << "]: ";
    for (int i = 0; i < t.size(); ++i)
    {
        strm << std::setw(10) << t[i] << " ";
    }
    return strm;
}

}  // namespace tinytorch
