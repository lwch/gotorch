#include <torch/torch.h>
#include "optimizer.h"

optimizer new_adam_optimizer(double lr, double beta1, double beta2, double eps, double weight_decay)
{
    auto options = torch::optim::AdamOptions(lr);
    options.betas(std::make_tuple(beta1, beta2));
    options.weight_decay(weight_decay);
    options.eps(eps);
    return new torch::optim::Adam(std::vector<torch::optim::OptimizerParamGroup>(), options);
}

void optimizer_step(optimizer optm, tensor *params, size_t params_count)
{
    std::vector<torch::Tensor> data;
    for (size_t i = 0; i < params_count; i++)
    {
        data.push_back(*params[i]);
    }
    std::vector<torch::optim::OptimizerParamGroup> &groups = optm->param_groups();
    groups.clear();
    torch::optim::OptimizerParamGroup group(data);
    group.set_options(optm->defaults().clone());
    groups.push_back(group);
    optm->step();
    optm->zero_grad();
}

double optimizer_get_lr(optimizer optm)
{
    return optm->defaults().get_lr();
}

void optimizer_set_lr(optimizer optm, double lr)
{
    optm->defaults().set_lr(lr);
}