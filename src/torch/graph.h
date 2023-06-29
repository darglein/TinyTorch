/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "ops.h"
#include "tensor.h"

#include <map>

#include "tiny_torch_config.h"

namespace TINY_TORCH_NAMESPACE
{
namespace autograd
{
struct Node;
}

// exact copy paste from pytorch
struct Edge
{
    Edge(std::shared_ptr<autograd::Node> function_, uint32_t input_nr_) noexcept
        : function(std::move(function_)), input_nr(input_nr_)
    {
    }

    /// The function this `Edge` points to.
    std::shared_ptr<autograd::Node> function;

    /// The identifier of a particular input to the function.
    uint32_t input_nr;
};


namespace autograd
{

struct IValue
{
    template <typename T>
    std::shared_ptr<T> toCustomClass()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
};

struct Context
{
    std::map<std::string, Tensor> data;
    std::map<std::string, int> data_int;
    // TODO: besser
    std::map<std::string, std::vector<int64_t>> data_sizes;

    std::map<std::string, IValue> saved_data;

    void set_materialize_grads(bool b) { throw std::runtime_error("not implemented"); }

    std::vector<Tensor> get_saved_variables()
    {
        throw std::runtime_error("not implemented");
        return {};
    }
    void save_for_backward(std::vector<Tensor> l) { throw std::runtime_error("not implemented"); }
};


struct Node
{
    // Create a node and give it a unique increasing sequence number
    Node() : sequence_nr(current_seq_nr++) {}
    virtual ~Node() {}

    // Computes and returns the gradients of the input tensor of the forward operator.
    // The input is the gradient of the forward output
    virtual std::vector<Tensor> backward(std::vector<Tensor> fwd_output_grad) = 0;

    // A global counter to get correct node ordering
    int sequence_nr;
    inline static int current_seq_nr;

    // The next edges are the inputs of the forward operator
    std::vector<std::shared_ptr<Edge>> next;

    // Variables that are required for the backward pass
    Context context;

    int64_t num_input_gradients_of_backward = 0;
};

using AutogradContext = Context;
using variable_list   = std::vector<Tensor>;


template <typename T>
struct FunctionNode : public Node
{
    FunctionNode() {}

    std::vector<Tensor> backward(std::vector<Tensor> fwd_output_grad) override
    {
        assert(fwd_output_grad.size() == num_input_gradients_of_backward);

        // backward
        auto grad_list = T::backward(context, fwd_output_grad);
        return grad_list;
    }

    static std::vector<Tensor> forward_and_build_graph(std::vector<Tensor> t)
    {
        // Create node and set next edge
        bool need_grad = false;
        auto node      = std::make_shared<FunctionNode<T>>();
        for (int i = 0; i < t.size(); ++i)
        {
            node->next.push_back(t[i].getEdge());
            // if(node->next.)
            if (t[i].requires_grad())
            {
                need_grad = true;
            }
        }

        // Forward
        auto result                           = T::forward(node->context, t);
        node->num_input_gradients_of_backward = result.size();

        // Set the edges of the output to point to this node
        for (int i = 0; i < result.size(); ++i)
        {
            result[i].set_requires_grad(need_grad);
            if (need_grad)
            {
                result[i].SetEdge(std::make_shared<Edge>(node, i));
            }
        }
        return result;
    }
};

template <typename T>
using Function = FunctionNode<T>;


struct AccumulateGrad : public Node
{
    AccumulateGrad(Tensor t) : t(t) { num_input_gradients_of_backward = 1; }

    std::vector<Tensor> backward(std::vector<Tensor> input_grad) override
    {
        assert(input_grad.size() == 1);
        // t.AddGradInplace(input_grad[0]);
        t.mutable_grad() += input_grad[0];
        return {};
    }

    Tensor t;
};
}  // namespace autograd

inline void MakeParameter(Tensor t)
{
    t.set_requires_grad(true);
    t.SetEdge(std::make_shared<Edge>(std::make_shared<autograd::AccumulateGrad>(t), 0));
}

struct NoGradGuard
{
    NoGradGuard() { throw std::runtime_error("not implemented"); }
};
struct AutoGradMode
{
    AutoGradMode(bool b) { throw std::runtime_error("not implemented"); }
};

}  // namespace TINY_TORCH_NAMESPACE