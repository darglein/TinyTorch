/**
 * Copyright (c) 2022 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "torch/tiny_torch.h"

int main()
{
    auto options = tinytorch::TensorOptions().device(tinytorch::kCUDA).dtype(tinytorch::kFloat);

    // The data tensors
    tinytorch::Tensor observation = tinytorch::rand({3, 5}, options);
    tinytorch::Tensor target      = tinytorch::rand({3, 5}, options);

    // The parameters of the model
    std::vector<tinytorch::Tensor> params;
    for (int i = 0; i < 4; ++i)
    {
        params.push_back(tinytorch::rand({3, 5}, options));
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
    tinytorch::optim::SGDOptimizer optim(params, 0.1);

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
