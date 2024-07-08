#include <torch/torch.h>
#include "exception.hpp"
#include "optimizer.h"

optimizer new_adam_optimizer(char **err, tensor *params, size_t params_count, double lr, double beta1, double beta2, double eps, double weight_decay)
{
    if (lr <= 0)
    {
        *err = strdup("lr must be positive");
        return nullptr;
    }
    if (beta1 < 0 || beta1 >= 1)
    {
        *err = strdup("beta1 must be a number between 0 and 1");
        return nullptr;
    }
    if (beta2 < 0 || beta2 >= 1)
    {
        *err = strdup("beta2 must be a number between 0 and 1");
        return nullptr;
    }
    if (eps <= 0)
    {
        *err = strdup("eps must be positive");
        return nullptr;
    }
    if (weight_decay < 0)
    {
        *err = strdup("weight_decay must be positive");
        return nullptr;
    }
    return auto_catch_optimizer([params, params_count, lr, beta1, beta2, eps, weight_decay]()
                                {
                                    auto options = torch::optim::AdamOptions(lr);
                                    options.betas(std::make_tuple(beta1, beta2));
                                    options.weight_decay(weight_decay);
                                    options.eps(eps);
                                    torch::autograd::variable_list data;
                                    for (size_t i = 0; i < params_count; i++)
                                    {
                                        data.push_back(*params[i]);
                                    }
                                    return new torch::optim::Adam(data, options); },
                                err);
}

optimizer new_adamw_optimizer(char **err, tensor *params, size_t params_count, double lr, double beta1, double beta2, double eps, bool amsgrad, double weight_decay)
{
    if (lr <= 0)
    {
        *err = strdup("lr must be positive");
        return nullptr;
    }
    if (beta1 < 0 || beta1 >= 1)
    {
        *err = strdup("beta1 must be a number between 0 and 1");
        return nullptr;
    }
    if (beta2 < 0 || beta2 >= 1)
    {
        *err = strdup("beta2 must be a number between 0 and 1");
        return nullptr;
    }
    if (eps <= 0)
    {
        *err = strdup("eps must be positive");
        return nullptr;
    }
    if (weight_decay < 0)
    {
        *err = strdup("weight_decay must be positive");
        return nullptr;
    }
    return auto_catch_optimizer([params, params_count, lr, beta1, beta2, eps, amsgrad, weight_decay]()
                                {
                                    auto options = torch::optim::AdamWOptions(lr);
                                    options.betas(std::make_tuple(beta1, beta2));
                                    options.weight_decay(weight_decay);
                                    options.eps(eps);
                                    options.amsgrad(amsgrad);
                                    torch::autograd::variable_list data;
                                    for (size_t i = 0; i < params_count; i++)
                                    {
                                        data.push_back(*params[i]);
                                    }
                                    return new torch::optim::AdamW(data, options); },
                                err);
}

void optimizer_step(char **err, optimizer optm)
{
    auto_catch_void([optm]()
                    {
                        optm->step();
                        optm->zero_grad(); },
                    err);
}

double optimizer_get_lr(char **err, optimizer optm)
{
    return auto_catch_double([optm]()
                             { return optm->defaults().get_lr(); },
                             err);
}

void optimizer_set_lr(char **err, optimizer optm, double lr)
{
    auto_catch_void([optm, lr]()
                    { optm->defaults().set_lr(lr); },
                    err);
}