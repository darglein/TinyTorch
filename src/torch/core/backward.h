#pragma once
#include "graph.h"
#include "torch/core/ops.h"
#include "torch/core/tensor.h"

namespace TINY_TORCH_NAMESPACE
{


void backward(Tensor loss)
{
    // for graph traversal
    std::vector<std::shared_ptr<autograd::Node>> node_stack;

    // Here we accumulate the gradients
    std::map<std::shared_ptr<autograd::Node>, std::vector<Tensor>> grad_map;


    assert(loss.numel() == 1);
    assert(loss.getEdge());

    // Start traversal at the root node
    auto root_node = loss.getEdge()->function;
    node_stack.push_back(root_node);

    // The gradient of the final loss is 1
    Tensor one = full({1}, 1);

    grad_map[root_node] = {one};

    while (!node_stack.empty())
    {
        // sort by sequence number
        std::sort(node_stack.begin(), node_stack.end(),
                  [](auto& n1, auto& n2) { return n1->sequence_nr < n2->sequence_nr; });

        // take the last node (which has the highest sequence number)
        auto current_node = node_stack.back();
        node_stack.pop_back();

        // backpropagate gradients
        auto next_gradients = current_node->node_backward(grad_map[current_node]);

        // Traverse to next nodes
        for (int i = 0; i < current_node->next.size(); ++i)
        {
            auto next = current_node->next[i];
            if (next)
            {
                auto next_node = next->function;

                // Accumulate gradient
                grad_map[next_node].resize(next_node->num_input_gradients_of_backward);

                if (!grad_map[next_node][next->input_nr].defined())
                {
                    grad_map[next_node][next->input_nr] = zeros_like(next_gradients[next->input_nr]);
                }
                grad_map[next_node][next->input_nr] += next_gradients[next->input_nr];

                // Add next node to the stack
                node_stack.push_back(next->function);
            }
        }
    }
}

}  // namespace TINY_TORCH_NAMESPACE