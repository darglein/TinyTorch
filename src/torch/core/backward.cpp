#include "backward.h"

#include "torch/core/graph.h"
#include "torch/core/ops/all.h"
namespace tinytorch
{



void backward(Tensor loss, Tensor grad, bool retain_grad)
{
    // for graph traversal
    std::vector<std::shared_ptr<autograd::Node>> node_stack;

    // Here we accumulate the gradients
    std::map<std::shared_ptr<autograd::Node>, std::vector<Tensor>> grad_map;


    CHECK(loss.requires_grad());
    CHECK(loss.getEdge());

    // Start traversal at the root node
    auto root_node = loss.getEdge()->function;
    node_stack.push_back(root_node);


    if (grad.defined())
    {
        CHECK_EQ(loss.device(), grad.device());
        CHECK_EQ(loss.sizes(), grad.sizes());
        grad_map[root_node] = {grad};
    }
    else
    {
        CHECK_EQ(loss.numel(), 1);
        // The gradient of the final loss is 1
        Tensor one          = full({1}, 1, loss.options());
        grad_map[root_node] = {one};
    }

    while (!node_stack.empty())
    {
        // sort by sequence number
        std::sort(node_stack.begin(), node_stack.end(),
                  [](auto& n1, auto& n2) { return n1->sequence_nr < n2->sequence_nr; });

        // remove duplicated nodes
        //  this can happen if one tensor is used multiple times
        auto end_it = std::unique(node_stack.begin(), node_stack.end());
        for (auto it = end_it; it != node_stack.end(); ++it)
        {
            autograd::Node* node = it->get();
            // CHECK_EQ(dynamic_cast<autograd::AccumulateGrad*>(node), nullptr);
        }
        node_stack.erase(end_it, node_stack.end());

        // take the last node (which has the highest sequence number)
        auto current_node = node_stack.back();
        node_stack.pop_back();


        // if (dynamic_cast<autograd::AccumulateGrad*>(current_node.get()))
        // {
        //     continue;
        // }

        // backpropagate gradients
        auto next_gradients = current_node->node_backward(grad_map[current_node]);

        CHECK_EQ(next_gradients.size(), current_node->num_inputs_of_forward);
        CHECK_EQ(current_node->next.size(), current_node->num_inputs_of_forward);

        // Traverse to next nodes
        for (int i = 0; i < current_node->next.size(); ++i)
        {
            auto next = current_node->next[i];
            if (next)
            {
                auto next_node = next->function;
                auto g         = next_gradients[i];
                // CHECK(g.defined());

                if (!g.defined())
                {
                    continue;
                }

                // Accumulate gradient
                grad_map[next_node].resize(next_node->num_input_gradients_of_backward);

                if (!grad_map[next_node][next->input_nr].defined())
                {
                    grad_map[next_node][next->input_nr] = g;
                }
                else
                {
                    grad_map[next_node][next->input_nr] += g;
                }

                // Add next node to the stack
                node_stack.push_back(next->function);
            }
        }
    }

    for (auto& it : grad_map)
    {
        autograd::AccumulateGrad* acc_node = dynamic_cast<autograd::AccumulateGrad*>(it.first.get());
        if (acc_node)
        {
            acc_node->accumulate(it.second);
            // acc_node->t.SetEdge(nullptr);
        }
    }

    if (1)
    {
        // clear graph
        for (auto& it : grad_map)
        {
            it.first->clear();
        }
    }
}
}  // namespace tinytorch