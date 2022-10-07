# TinyTorch
A Minimalistic Auto-Diff Optimization Framework for Teaching and Understanding Pytorch


### This Project is for

:heavy_check_mark: Teaching Auto-Diff and Backpropagation

:heavy_check_mark: Introducing or Learning libTorch/Pytorch

:heavy_check_mark: Showing that Auto-Diff Optimization is not Magic


### This Project is NOT for

:heavy_multiplication_x: Fitting Models to your own Data

:heavy_multiplication_x: Replacing Pytorch or any other Optimization Framework


# Features of TinyTorch

* Forward and Backward Implementations of basic Operations (+, -, *, sum, square)
* Automatic Graph Generation during forward
* Backpropagation through the generated graph
* Almost the same syntax as libTorch


## Example Code

```c++
#include "tiny_torch.h"

int main()
{
    // The data tensors
    tinytorch::Tensor observation = tinytorch::rand(10);
    tinytorch::Tensor target      = tinytorch::rand(10);

    // The parameters of the model
    std::vector<tinytorch::Tensor> params;
    for (int i = 0; i < 4; ++i)
    {
        params.push_back(tinytorch::rand(10));
        MakeParameter(params.back());
    }

    // The model itself
    auto model = [&](tinytorch::Tensor x) -> tinytorch::Tensor
    {
        x = x * params[0];
        x = x + params[1];
        x = x * params[2];
        x = x + params[3];
        return x;
    };

    tinytorch::RMSPropOptimizer optim(params, 0.1);
    for (int i = 0; i < 100; ++i)
    {
        optim.ZeroGrad();

        auto prediction = model(observation);

        auto loss = sum(square(prediction - target));

        backward(loss);
        optim.Step();
        std::cout << "Step " << i << " Loss: " << loss << std::endl;
    }
    return 0;
}

```

