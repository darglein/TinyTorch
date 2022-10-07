/**
* Copyright (c) 2022 Darius Rückert
* Licensed under the MIT License.
* See LICENSE file for more information.
*/

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

    tinytorch::RMSPropOptimizer optim(params, 0.01);
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
