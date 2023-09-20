#include <torch/torch.h>
#include "exception.hpp"
#include "operator.h"

void tensor_backward(char **err, tensor a, bool retain)
{
    auto_catch_void([a, retain]()
                    {
                        torch::Tensor gradient;
                        a->backward(gradient, retain); },
                    err);
}

tensor tensor_matmul(char **err, tensor a, tensor b)
{
    return auto_catch_tensor([a, b]()
                             { return new torch::Tensor(a->matmul(*b)); },
                             err);
}

tensor tensor_add(char **err, tensor a, tensor b)
{
    return auto_catch_tensor([a, b]()
                             { return new torch::Tensor(a->add(*b)); },
                             err);
}

tensor tensor_sub(char **err, tensor a, tensor b)
{
    return auto_catch_tensor([a, b]()
                             { return new torch::Tensor(a->sub(*b)); },
                             err);
}

tensor tensor_mul(char **err, tensor a, tensor b)
{
    return auto_catch_tensor([a, b]()
                             { return new torch::Tensor(a->mul(*b)); },
                             err);
}

tensor tensor_div(char **err, tensor a, tensor b)
{
    return auto_catch_tensor([a, b]()
                             { return new torch::Tensor(a->div(*b)); },
                             err);
}

tensor tensor_pow(char **err, tensor a, double n)
{
    return auto_catch_tensor([a, n]()
                             { return new torch::Tensor(a->pow(n)); },
                             err);
}

tensor tensor_sqrt(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->sqrt()); },
                             err);
}

tensor tensor_rsqrt(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->rsqrt()); },
                             err);
}

tensor tensor_log(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->log()); },
                             err);
}

tensor tensor_exp(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->exp()); },
                             err);
}

tensor tensor_neg(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->neg()); },
                             err);
}

tensor tensor_abs(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->abs()); },
                             err);
}

tensor tensor_max(char **err, tensor a, int64_t dim, bool keepdim)
{
    return auto_catch_tensor([a, dim, keepdim]()
                             { return new torch::Tensor(std::get<0>(a->max(dim, keepdim))); },
                             err);
}

tensor tensor_min(char **err, tensor a, int64_t dim, bool keepdim)
{
    return auto_catch_tensor([a, dim, keepdim]()
                             { return new torch::Tensor(std::get<0>(a->min(dim, keepdim))); },
                             err);
}

tensor tensor_sum(char **err, tensor a, int64_t dim, bool keepdim)
{
    return auto_catch_tensor([a, dim, keepdim]()
                             { return new torch::Tensor(a->sum(dim, keepdim)); },
                             err);
}

tensor tensor_mean(char **err, tensor a, int64_t dim, bool keepdim)
{
    return auto_catch_tensor([a, dim, keepdim]()
                             { return new torch::Tensor(a->mean(dim, keepdim)); },
                             err);
}

tensor tensor_var(char **err, tensor a, int64_t dim, bool unbiased, bool keepdim)
{
    return auto_catch_tensor([a, dim, unbiased, keepdim]()
                             { return new torch::Tensor(a->var(dim, unbiased, keepdim)); },
                             err);
}

tensor tensor_relu(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->relu()); },
                             err);
}

tensor tensor_gelu(char **err, tensor a, bool tanh)
{
    return auto_catch_tensor([a, tanh]()
                             {
                                if (tanh)
                                {
                                    return new torch::Tensor(torch::gelu(*a, "tanh"));
                                }
                                return new torch::Tensor(torch::gelu(*a)); },
                             err);
}

tensor tensor_sigmoid(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->sigmoid()); },
                             err);
}

tensor tensor_tanh(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->tanh()); },
                             err);
}

tensor tensor_softmax(char **err, tensor a, int64_t dim)
{
    return auto_catch_tensor([a, dim]()
                             { return new torch::Tensor(a->softmax(dim)); },
                             err);
}

tensor tensor_softmax1(char **err, tensor a, int64_t dim)
{
    return auto_catch_tensor([a, dim]()
                             {
                                torch::Tensor x = *a - std::get<0>(a->max(dim, true));
                                torch::Tensor exp_x = torch::exp(x);
                                return new torch::Tensor(exp_x / (1 + exp_x.sum(dim,true))); },
                             err);
}

tensor tensor_dropout(char **err, tensor a, double p, bool train)
{
    return auto_catch_tensor([a, p, train]()
                             { return new torch::Tensor(torch::dropout(*a, p, train)); },
                             err);
}

tensor tensor_unsqueeze(char **err, tensor a, int64_t dim)
{
    return auto_catch_tensor([a, dim]()
                             { return new torch::Tensor(a->unsqueeze(dim)); },
                             err);
}

tensor tensor_squeeze(char **err, tensor a, int64_t dim)
{
    return auto_catch_tensor([a, dim]()
                             { return new torch::Tensor(a->squeeze(dim)); },
                             err);
}

tensor tensor_contiguous(char **err, tensor a)
{
    return auto_catch_tensor([a]()
                             { return new torch::Tensor(a->contiguous()); },
                             err);
}

tensor tensor_expand(char **err, tensor a, int64_t *sizes, size_t len)
{
    return auto_catch_tensor([a, sizes, len]()
                             { return new torch::Tensor(a->expand(torch::IntArrayRef(sizes, len))); },
                             err);
}