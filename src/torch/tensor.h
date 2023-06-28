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

#include "tiny_torch_config.h"

namespace tinytorch
{

struct Edge;
struct Tensor;
struct TensorImpl;


struct TINYTORCH_API Tensor
{
    Tensor(int size = 0);
    Tensor(std::vector<float> data);
    int size() const;
    float& operator[](int idx);
    void resize(int new_size);


    void ClearGrad();
    const Tensor& grad();
    Tensor& mutable_grad();

    void AddGradInplace(Tensor g);
    void AddInplace(Tensor g);
    std::shared_ptr<Edge> getEdge();
    void SetEdge(std::shared_ptr<Edge> edge);


    void set_requires_grad(bool requires_grad);
    bool requires_grad();

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
    TensorImpl(int size) : data(size) {}
    TensorImpl(std::vector<float> data) : data(data) {}


    void set_requires_grad(bool requires_grad)
    {
        if (requires_grad)
        {
            autograd_meta = std::make_unique<AutogradMeta>();
        }
        else
        {
            autograd_meta = nullptr;
        }
    }

    bool requires_grad()
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


    std::vector<float> data;
    std::unique_ptr<AutogradMeta> autograd_meta;
    // required for .backward()
    // std::vector<float> grad;
};


}  // namespace tinytorch
