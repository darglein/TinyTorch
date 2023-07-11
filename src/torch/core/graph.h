/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "torch/core/ops.h"
#include "torch/core/tensor.h"

#include <map>

#include "torch/tiny_torch_config.h"

namespace tinytorch
{

template <typename _Tp, typename... _Args>
std::shared_ptr<_Tp> make_intrusive(_Args&&... __args)
{
    return std::make_shared<_Tp>(std::forward<_Args>(__args)...);
}

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


struct IValue
{
    IValue() {}

    IValue(bool b) : v_bool(b) {}
    IValue(double d) :v_double(d) {}
    IValue(int32_t i) : v_int64(i) {}
    IValue(int64_t i) : v_int64(i) {}

    template <typename T>
    IValue(std::shared_ptr<T> i) : custom_class(i) {}

    template <typename T>
    std::shared_ptr<T> toCustomClass()
    {
        auto result = std::dynamic_pointer_cast<T>(custom_class);
        CHECK(result);
        return result;
    }

    bool toBool() { return v_bool; }
    double toDouble() { return v_double; }
    int64_t toInt() { return v_int64; }

    bool v_bool;
    double v_double;
    int64_t v_int64;
    std::shared_ptr<CustomClassHolder> custom_class;
};


namespace autograd
{

struct Context
{
    std::map<std::string, int> data_int;
    // TODO: besser
    std::map<std::string, SizeType> data_sizes;

    std::map<std::string, IValue> saved_data;

    std::vector<Tensor> saved_tensors;

    void set_materialize_grads(bool b) { throw std::runtime_error("not implemented"); }

    std::vector<Tensor> get_saved_variables() { return saved_tensors; }
    void save_for_backward(const std::vector<Tensor>& l) { saved_tensors = l; }
};


using AutogradContext = Context;
using variable_list   = std::vector<Tensor>;
using Variable        = Tensor;


struct ExtractVariables
{
    template <typename T, typename... Args>
    static void apply(std::vector<Tensor>& output, T&& arg, Args&&... args)
    {
        check_tensor(output, arg);
        apply(output, args...);
    }

    static void apply(std::vector<Tensor>& output) {}

    template <typename T>
    static void check_tensor(std::vector<Tensor>& output, const T& arg)
    {
    }

    static void check_tensor(std::vector<Tensor>& output, const Tensor& arg) { output.push_back(arg); }
};


struct Node
{
    // Create a node and give it a unique increasing sequence number
    Node() : sequence_nr(current_seq_nr++) {}
    virtual ~Node() {}

    // Computes and returns the gradients of the input tensor of the forward operator.
    // The input is the gradient of the forward output
    virtual std::vector<Tensor> node_backward(const std::vector<Tensor>& fwd_output_grad) = 0;

    // A global counter to get correct node ordering
    int sequence_nr;
    inline static int current_seq_nr;

    // The next edges are the inputs of the forward operator
    std::vector<std::shared_ptr<Edge>> next;

    // Variables that are required for the backward pass
    Context context;

    int64_t num_input_gradients_of_backward = 0;
    int64_t num_inputs_of_forward           = 0;
};


template <typename T>
struct FunctionNode : public Node
{
    FunctionNode() {}

    std::vector<Tensor> node_backward(const std::vector<Tensor>& fwd_output_grad) override
    {
        CHECK_EQ(fwd_output_grad.size(), num_input_gradients_of_backward);

        // backward
        auto grad_list = T::backward(&context, fwd_output_grad);
        CHECK_EQ(grad_list.size(), num_inputs_of_forward);
        return grad_list;
    }

    template <typename... Args>
    static std::vector<Tensor> forward_and_build_graph(Args&&... args)
    {
        // Create node and set next edge
        bool need_grad              = false;
        auto node                   = std::make_shared<FunctionNode<T>>();
        node->num_inputs_of_forward = sizeof...(Args);

        std::vector<Tensor> t;
        ExtractVariables::apply(t, args...);
        // CHECK_EQ(t.size() , num_inputs);
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
        auto result                           = T::forward(&node->context, std::forward<Args>(args)...);
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

    template <typename... Args>
    static std::vector<Tensor> apply(Args&&... args)
    {
        return forward_and_build_graph(args...);
    }
};

template <typename T>
using Function = FunctionNode<T>;


struct AccumulateGrad : public Node
{
    AccumulateGrad(Tensor t) : t(t) { num_input_gradients_of_backward = 1; }

    std::vector<Tensor> node_backward(const std::vector<Tensor>& input_grad) override
    {
        CHECK_EQ(input_grad.size(), 1);
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

}  // namespace tinytorch