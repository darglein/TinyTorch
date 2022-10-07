#pragma once

#include "tensor.h"

#include <map>

#define GRAPH_STEP -1

namespace tinytorch
{

struct Node;

// exact copy paste from pytorch
struct Edge
{
    Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
        : function(std::move(function_)), input_nr(input_nr_)
    {
    }

    /// The function this `Edge` points to.
    std::shared_ptr<Node> function;

    /// The identifier of a particular input to the function.
    uint32_t input_nr;
};

struct Context
{
    std::map<std::string, Tensor> data;
    std::map<std::string, int> data_int;
};

struct Node
{
    // Create a node and give it a unique increasing sequence number
    Node() : sequence_nr(current_seq_nr++) {}

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

    int num_inputs;
};



template <typename T>
struct FunctionNode : public Node
{
    FunctionNode() {}

    std::vector<Tensor> backward(std::vector<Tensor> fwd_output_grad) override
    {
        // backward
        auto grad_list = T::backward(context, fwd_output_grad);
        return grad_list;
    }

    static std::vector<Tensor> forward_and_build_graph(std::vector<Tensor> t)
    {
        // Create node and set next edge
        auto node = std::make_shared<FunctionNode<T>>();
        for (int i = 0; i < t.size(); ++i)
        {
            node->next.push_back(t[i].getEdge());
        }
        node->num_inputs = t.size();


        // Forward
        auto result = T::forward(node->context, t);

        // Set the edges of the output to point to this node
        for (int i = 0; i < result.size(); ++i)
        {
            result[i].SetEdge(std::make_shared<Edge>(node, i));
        }
        return result;
    }
};



struct AccumulateGrad : public Node
{
    AccumulateGrad(Tensor t) : t(t) { num_inputs = 1; }

    std::vector<Tensor> backward(std::vector<Tensor> input_grad) override
    {
        assert(input_grad.size() == 1);
        t.AddGradInplace(input_grad[0]);
        return {};
    }

    Tensor t;
};

inline void MakeParameter(Tensor t)
{
    t.SetEdge(std::make_shared<Edge>(std::make_shared<AccumulateGrad>(t), 0));
}

}  // namespace tinytorch