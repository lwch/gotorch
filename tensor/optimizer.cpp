#include <torch/torch.h>
#include "optimizer.h"

optimizer new_adam_optimizer(double lr, double beta1, double beta2, double weight_decay)
{
    auto options = torch::optim::AdamOptions(lr);
    options.betas(std::make_tuple(beta1, beta2));
    options.weight_decay(weight_decay);
    return new torch::optim::Adam(std::vector<torch::optim::OptimizerParamGroup>(), options);
}

void optimizer_step(optimizer optm, tensor *params, size_t params_count)
{
    std::vector<torch::Tensor> &params_vec = optm->parameters();
    params_vec.clear();
    for (size_t i = 0; i < params_count; i++)
    {
        params_vec.push_back(*params[i]);
    }
    optm->zero_grad();
    optm->step();
}