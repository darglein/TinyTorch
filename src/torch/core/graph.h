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

template <typename T>
using intrusive_ptr = std::shared_ptr<T>;

struct TINYTORCH_API GradMode
{
    static bool is_enabled();
    static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct TINYTORCH_API AutoGradMode
{
    AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) { GradMode::set_enabled(enabled); }
    ~AutoGradMode() { GradMode::set_enabled(prev_mode); }
    bool prev_mode;
};

struct NoGradGuard : public AutoGradMode
{
    NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};


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
struct TINYTORCH_API Edge
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


struct TINYTORCH_API IValue
{
    IValue() {}

    IValue(bool b) : v_bool(b) {}
    IValue(double d) : v_double(d) {}
    IValue(int32_t i) : v_int64(i) {}
    IValue(int64_t i) : v_int64(i) {}
    IValue(Tensor t) : v_tensor(t) {}
    IValue(SizeType s) : v_size(s) {}
    IValue(Device d) : v_device(d) {}

    template <typename T>
    IValue(std::shared_ptr<T> i) : custom_class(i)
    {
    }

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
    Tensor toTensor() { return v_tensor; }
    SizeType toSizes() { return v_size; }
    Device toDevice() { return v_device; }

    bool v_bool;
    double v_double;
    int64_t v_int64;
    Tensor v_tensor;
    SizeType v_size;
    Device v_device;
    std::shared_ptr<CustomClassHolder> custom_class;
};


namespace autograd
{

struct TensorMetaData
{
    SizeType size;
    Device device;
    bool requires_grad = false;
};

struct TINYTORCH_API Context
{
    // the input tensor dimensions
    // is used to check for the correct gradient size
    std::vector<TensorMetaData> next_meta;


    std::map<std::string, int> data_int;

    std::map<std::string, IValue> saved_data;

    std::vector<Tensor> saved_tensors;

    bool requires_grad = true;

    bool requires_grad_for_input(int id) const
    {
        CHECK_LT(id, next_meta.size());
        return next_meta[id].requires_grad;
    }

    // Sets whether undefined output grad tensors should be expanded to tensors
    // full of zeros before calling backward function. Default value is true.
    void set_materialize_grads(bool b) {}

    std::vector<Tensor> get_saved_variables() { return saved_tensors; }
    void save_for_backward(const std::vector<Tensor>& l) { saved_tensors = l; }
};


using AutogradContext = Context;
using variable_list   = std::vector<Tensor>;
using Variable        = Tensor;


template <typename TargetT>
struct ExtractVariables
{
    template <typename T, typename... Args>
    static void apply(std::vector<TargetT>& output, T&& arg, Args&&... args)
    {
        check_tensor(output, arg);
        apply(output, args...);
    }

    static void apply(std::vector<TargetT>& output) {}

    template <typename T>
    static void check_tensor(std::vector<TargetT>& output, const T& arg)
    {
    }

    static void check_tensor(std::vector<TargetT>& output, const TargetT& arg) { output.push_back(arg); }
};


struct ToIValueList
{
    template <typename T, typename... Args>
    static void apply(std::vector<IValue>& output, T&& arg, Args&&... args)
    {
        output.push_back(arg);
        apply(output, args...);
    }

    static void apply(std::vector<IValue>& output) {}
};

struct NodeInfo
{
    NodeInfo(std::string n, Context* c) : name(n), context(c) {}
    std::string name;
    Context* context;
};
struct NodeCallback
{
    virtual ~NodeCallback() {}
    virtual void ForwardStart(const NodeInfo& info) = 0;
    virtual void ForwardEnd(const NodeInfo& info)   = 0;

    virtual void BackwardStart(const NodeInfo& info) = 0;
    virtual void BackwardEnd(const NodeInfo& info)   = 0;
};


extern TINYTORCH_API NodeCallback* call_back;

struct FunctionCallbackScope
{
    FunctionCallbackScope(std::string function_name) : function_name(function_name)
    {
        if (call_back) call_back->ForwardStart(NodeInfo(function_name, &context));
    }
    ~FunctionCallbackScope()
    {
        if (call_back) call_back->ForwardEnd(NodeInfo(function_name, &context));
    }
    std::string function_name;
    Context context;
};

#define TINYTORCH_LOG_FUNCTION_CALL() auto f = autograd::FunctionCallbackScope(TT_FUNCTION_NAME)

struct TINYTORCH_API Node
{
    // Create a node and give it a unique increasing sequence number
    Node();
    virtual ~Node() {}

    void clear()
    {
        next.clear();
        context = Context();
    }
    // Computes and returns the gradients of the input tensor of the forward operator.
    // The input is the gradient of the forward output
    virtual std::vector<Tensor> node_backward(const std::vector<Tensor>& fwd_output_grad) = 0;

    // A global counter to get correct node ordering
    int64_t sequence_nr;

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
        NoGradGuard ngg;
        CHECK_EQ(fwd_output_grad.size(), num_input_gradients_of_backward);


        if (call_back) call_back->BackwardStart(NodeInfo(std::string("bw_") + typeid(T).name(), &context));

        CHECK(context.requires_grad);
        // backward
        auto grad_list = T::backward(&context, fwd_output_grad);
        CHECK_EQ(grad_list.size(), num_inputs_of_forward);
        CHECK_EQ(context.next_meta.size(), num_inputs_of_forward);

        // the gradient produced by the backward function must be either undefined
        // or have the same dimensions as the input tensor
        for (int i = 0; i < grad_list.size(); ++i)
        {
            if (grad_list[i].defined())
            {
                if (!(grad_list[i].sizes() == context.next_meta[i].size))
                {
                    std::cerr << "incorrect gradient size found for node " << typeid(FunctionNode<T>).name()
                              << std::endl;
                    for (int j = 0; j < fwd_output_grad.size(); ++j)
                    {
                        std::cerr << "fwd_output_grad " << fwd_output_grad[j].sizes() << std::endl;
                    }
                    for (int j = 0; j < grad_list.size(); ++j)
                    {
                        std::cerr << "input " << context.next_meta[j].size << " grad ";
                        if (grad_list[j].defined())
                        {
                            std::cerr << grad_list[j].sizes();
                        }
                        else
                        {
                            std::cerr << "[undefined]";
                        }
                        std::cerr << "\n";
                    }

                    CHECK(false) << "incorrect gradient size";
                }
            }
        }

        if (call_back) call_back->BackwardEnd(NodeInfo(std::string("bw_") + typeid(T).name(), &context));

        // This is important, because if the node would store the output tensor here then
        // we get a shared_ptr cycle and leak memory.
        context.saved_tensors.clear();

        return grad_list;
    }

    template <typename... Args>
    static std::vector<Tensor> forward_and_build_graph(Args&&... args)
    {
        // Create node and set next edge
        bool need_grad              = false;
        auto node                   = std::make_shared<FunctionNode<T>>();
        node->num_inputs_of_forward = sizeof...(Args);

        if (GradMode::is_enabled())
        {
            std::vector<IValue> input_ivalues;
            ToIValueList::apply(input_ivalues, args...);
            CHECK_EQ(input_ivalues.size(), node->num_inputs_of_forward);
            for (int i = 0; i < input_ivalues.size(); ++i)
            {
                if (input_ivalues[i].v_tensor.defined())
                {
                    auto t = input_ivalues[i].toTensor();
                    node->next.push_back(t.getEdge());

                    TensorMetaData meta;
                    meta.size   = t.sizes();
                    meta.device = t.device();
                    if (t.requires_grad())
                    {
                        need_grad          = true;
                        meta.requires_grad = true;
                    }
                    node->context.next_meta.push_back(meta);
                }
                else
                {
                    node->next.push_back({});
                    node->context.next_meta.push_back({});
                }
            }
            CHECK_EQ(node->next.size(), node->num_inputs_of_forward);
        }

        node->context.requires_grad = need_grad;


        if (call_back) call_back->ForwardStart(NodeInfo(typeid(T).name(), &node->context));

        // Forward
        NoGradGuard ngg;
        auto result                           = T::forward(&node->context, std::forward<Args>(args)...);
        node->num_input_gradients_of_backward = result.size();

        // Set the edges of the output to point to this node
        for (int i = 0; i < result.size(); ++i)
        {
            CHECK(result[i].defined());
            result[i].set_requires_grad(need_grad);
            if (need_grad)
            {
                result[i].SetEdge(std::make_shared<Edge>(node, i));
            }
        }

        if (call_back) call_back->ForwardEnd(NodeInfo(typeid(T).name(), &node->context));
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
    AccumulateGrad(std::shared_ptr<TensorImpl> t);

    std::vector<Tensor> node_backward(const std::vector<Tensor>& input_grad) override;

    std::vector<Tensor> accumulate(const std::vector<Tensor>& input_grad);

    // the weak ptr is important because otherwise we get a cyclic dependency to the owning tensor
    std::weak_ptr<TensorImpl> impl_;
};
}  // namespace autograd

}  // namespace tinytorch