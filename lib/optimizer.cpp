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

static void *build_optimizer_param_state(optimizer optm, void *ptr)
{
    {
        torch::optim::Adam *op = dynamic_cast<torch::optim::Adam *>(optm);
        if (op)
        {
            optm->state()[ptr] = std::make_unique<torch::optim::AdamParamState>();
            return optm->state()[ptr].get();
        }
    }
    {
        torch::optim::AdamW *op = dynamic_cast<torch::optim::AdamW *>(optm);
        if (op)
        {
            optm->state()[ptr] = std::make_unique<torch::optim::AdamWParamState>();
            return optm->state()[ptr].get();
        }
    }
    return nullptr;
}

optimizer_state optimizer_get_state(char **err, optimizer optm)
{
    return auto_catch_optimizer_state([optm]()
                                      {
                                        std::vector<void*> params;
                                        for (auto group : optm->param_groups())
                                        {
                                            for (auto p : group.params())
                                                params.push_back(p.unsafeGetTensorImpl());
                                        }
                                        std::vector<torch::optim::OptimizerParamState*> tmp;
                                        for (auto p: params)
                                        {
                                            torch::optim::OptimizerParamState* ptr = optm->state()[p].get();
                                            if (!ptr) {
                                                ptr = reinterpret_cast<torch::optim::OptimizerParamState*>(build_optimizer_param_state(optm, p));
                                            }
                                            tmp.push_back(optm->state()[p].get());
                                        }
                                        return new _optimizer_state{tmp}; },
                                      err);
}

void optimizer_state_free(optimizer_state state)
{
    delete state;
}

size_t optimizer_state_count(char **err, optimizer_state state)
{
    return auto_catch_size_t([state]()
                             { return state->data.size(); },
                             err);
}

size_t optimizer_state_size(char **err, optimizer_state state, size_t index)
{
    return auto_catch_size_t([state, index]()
                             {
                                torch::optim::OptimizerParamState* ptr = state->data[index];
                                {
                                    torch::optim::AdamParamState* p = dynamic_cast<torch::optim::AdamParamState*>(ptr);
                                    if (p) {
                                        if (p->max_exp_avg_sq().numel()) return 4;
                                        return 3;
                                    }
                                }
                                {
                                    torch::optim::AdamWParamState* p = dynamic_cast<torch::optim::AdamWParamState*>(ptr);
                                    if (p) {
                                        if (p->max_exp_avg_sq().numel()) return 4;
                                        return 3;
                                    }
                                }
                                return 0; },
                             err);
}

template <typename T>
tensor get_adam_state(T ptr, size_t key)
{
    switch (key)
    {
    case 0:
        return new torch::Tensor(torch::tensor({ptr->step()}));
    case 1:
        return new torch::Tensor(ptr->exp_avg());
    case 2:
        return new torch::Tensor(ptr->exp_avg_sq());
    case 3:
        return new torch::Tensor(ptr->max_exp_avg_sq());
    }
    return nullptr;
}

tensor optimizer_state_get(char **err, optimizer_state state, size_t index, size_t key)
{
    return auto_catch_tensor([state, index, key]()
                             {
                                torch::optim::OptimizerParamState* ptr = state->data[index];
                                {
                                    torch::optim::AdamParamState *p = dynamic_cast<torch::optim::AdamParamState*>(ptr);
                                    if (p) return get_adam_state(p, key);
                                }
                                {
                                    torch::optim::AdamWParamState *p = dynamic_cast<torch::optim::AdamWParamState*>(ptr);
                                    if (p) return get_adam_state(p, key);
                                }
                                return new torch::Tensor(); },
                             err);
}

template <typename T>
void set_adam_state(T ptr, size_t key, tensor value)
{
    switch (key)
    {
    case 0:
        ptr->step(value->item<int64_t>());
        break;
    case 1:
        ptr->exp_avg(*value);
        break;
    case 2:
        ptr->exp_avg_sq(*value);
        break;
    case 3:
        ptr->max_exp_avg_sq(*value);
        break;
    }
}

void optimizer_state_set(char **err, optimizer_state state, size_t index, size_t key, tensor value)
{
    auto_catch_void([state, index, key, value]()
                    {
                        torch::optim::OptimizerParamState* ptr = state->data[index];
                        {
                            torch::optim::AdamParamState *p = dynamic_cast<torch::optim::AdamParamState*>(ptr);
                            if (p)
                            {
                                set_adam_state(p, key, value);
                                return;
                            }
                        }
                        {
                            torch::optim::AdamWParamState *p = dynamic_cast<torch::optim::AdamWParamState*>(ptr);
                            if (p)
                            {
                                set_adam_state(p, key, value);
                                return;
                            }
                        } },
                    err);
}