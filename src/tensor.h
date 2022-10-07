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


}  // namespace tinytorch
