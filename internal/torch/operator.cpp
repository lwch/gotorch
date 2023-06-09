#include <torch/torch.h>
#include "operator.h"

void tensor_backward(tensor a)
{
    a->backward();
}

tensor tensor_matmul(tensor a, tensor b)
{
    return new torch::Tensor(a->matmul(*b));
}

tensor tensor_add(tensor a, tensor b)
{
    return new torch::Tensor(a->add(*b));
}

tensor tensor_sub(tensor a, tensor b)
{
    return new torch::Tensor(a->sub(*b));
}

tensor tensor_mul(tensor a, tensor b)
{
    return new torch::Tensor(a->mul(*b));
}

tensor tensor_div(tensor a, tensor b)
{
    return new torch::Tensor(a->div(*b));
}

tensor tensor_pow(tensor a, double n)
{
    return new torch::Tensor(a->pow(n));
}

tensor tensor_sqrt(tensor a)
{
    return new torch::Tensor(a->sqrt());
}

tensor tensor_log(tensor a)
{
    return new torch::Tensor(a->log());
}

tensor tensor_exp(tensor a)
{
    return new torch::Tensor(a->exp());
}

tensor tensor_neg(tensor a)
{
    return new torch::Tensor(a->neg());
}

tensor tensor_abs(tensor a)
{
    return new torch::Tensor(a->abs());
}

tensor tensor_max(tensor a, int64_t dim, bool keepdim)
{
    return new torch::Tensor(std::get<0>(a->max(dim, keepdim)));
}

tensor tensor_min(tensor a, int64_t dim, bool keepdim)
{
    return new torch::Tensor(std::get<0>(a->min(dim, keepdim)));
}

tensor tensor_sum(tensor a, int64_t dim, bool keepdim)
{
    return new torch::Tensor(a->sum(dim, keepdim));
}

tensor tensor_mean(tensor a, int64_t dim, bool keepdim)
{
    return new torch::Tensor(a->mean(dim, keepdim));
}

tensor tensor_relu(tensor a)
{
    return new torch::Tensor(a->relu());
}

tensor tensor_sigmoid(tensor a)
{
    return new torch::Tensor(a->sigmoid());
}

tensor tensor_tanh(tensor a)
{
    return new torch::Tensor(a->tanh());
}

tensor tensor_softmax(tensor a, int64_t dim)
{
    return new torch::Tensor(a->softmax(dim));
}

tensor tensor_dropout(tensor a, double p, bool train)
{
    return new torch::Tensor(torch::dropout(*a, p, train));
}