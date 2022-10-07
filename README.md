# TinyTorch
A Minimalistic Auto-Diff Optimization Framework for Teaching and Understanding Pytorch


### This Project is for

:heavy_check_mark: Teaching auto-diff and backpropagation

:heavy_check_mark: Introducing or learning libTorch/Pytorch

:heavy_check_mark: Showing that auto-diff optimization is not magic


### This Project is NOT for

:heavy_multiplication_x: Fitting models to your own data

:heavy_multiplication_x: Replacing Pytorch or any other optimization framework


# Features of TinyTorch

* Forward and backward implementations of basic operations (+, -, *, sum, square)
* Automatic graph generation during forward
* Backpropagation through the generated graph
* Almost the same syntax as libTorch
* No dependencies except the std-library

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

    // Create a simple optimizer
    tinytorch::SGDOptimizer optim(params, 0.1);

    // Optimize the model for 50 iterations
    for (int i = 0; i < 50; ++i)
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

Example Output:
```
Step 0 Loss: [Tensor s=1]:    5.73343 
Step 1 Loss: [Tensor s=1]:    2.18133 
Step 2 Loss: [Tensor s=1]:    1.17362 
Step 3 Loss: [Tensor s=1]:   0.321065 
Step 4 Loss: [Tensor s=1]:  0.0889282 
Step 5 Loss: [Tensor s=1]:  0.0313245 
Step 6 Loss: [Tensor s=1]:  0.0113702 
Step 7 Loss: [Tensor s=1]:  0.0036413 
Step 8 Loss: [Tensor s=1]: 0.00149557 
Step 9 Loss: [Tensor s=1]: 0.000541403 
Step 10 Loss: [Tensor s=1]: 0.000217643 
Step 11 Loss: [Tensor s=1]: 8.67411e-05 
Step 12 Loss: [Tensor s=1]: 3.2838e-05 
Step 13 Loss: [Tensor s=1]: 1.26026e-05 
...
```

