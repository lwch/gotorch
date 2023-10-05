#include <torch/torch.h>
#include "exception.hpp"
#include "tensor.h"

tensor scaled_dot_product_attention(char **err,
                                    tensor q, tensor k, tensor v,
                                    tensor mask, double dropout, bool is_causal)
{
    return auto_catch_tensor([q, k, v, mask, dropout, is_causal]()
                             {
                                c10::optional<at::Tensor> mask_opt = c10::nullopt;
                                if (mask) {
                                    mask_opt = *mask;
                                }
                                return new torch::Tensor(
                                    torch::scaled_dot_product_attention(*q, *k, *v, mask_opt, dropout, is_causal)); },
                             err);
}

void clip_grad_norm(char **err,
                    tensor *params, size_t params_count,
                    double max_norm, double norm_type)
{
    auto_catch_void([params, params_count, max_norm, norm_type]()
                    {
                        std::vector<torch::Tensor> list;
                        for (size_t i=0;i<params_count;i++)
                        {
                            list.push_back(*params[i]);
                        }
                        torch::nn::utils::clip_grad_norm_(list, max_norm, norm_type); },
                    err);
}

void tensor_print(tensor t)
{
    t->print();
}

tensor tensor_cat(char **err, tensor *tensors, size_t tensors_len, int64_t dim)
{
    return auto_catch_tensor([tensors, tensors_len, dim]()
                             {
                                 std::vector<torch::Tensor> list;
                                 for (size_t i = 0; i < tensors_len; i++)
                                 {
                                     list.push_back(*tensors[i]);
                                 }
                                 return new torch::Tensor(torch::cat(list, dim)); },
                             err);
}

tensor tensor_embedding(char **err, tensor weight, tensor indices, int64_t padding_idx)
{
    return auto_catch_tensor([weight, indices, padding_idx]()
                             { return new torch::Tensor(torch::embedding(*weight, *indices, padding_idx)); },
                             err);
}

void init_kaiming_uniform(char **err, tensor t, double a)
{
    return auto_catch_void([t, a]()
                           { torch::nn::init::kaiming_uniform_(*t, a); },
                           err);
}

void init_xaiver_uniform(char **err, tensor t, double gain)
{
    return auto_catch_void([t, gain]()
                           { torch::nn::init::xavier_uniform_(*t, gain); },
                           err);
}

void init_normal(char **err, tensor t, double mean, double std)
{
    return auto_catch_void([t, mean, std]()
                           { torch::nn::init::normal_(*t, mean, std); },
                           err);
}

void init_zeros(char **err, tensor t)
{
    return auto_catch_void([t]()
                           { torch::nn::init::zeros_(*t); },
                           err);
}

void svd(char **err, tensor t, tensor *u, tensor *s, tensor *v)
{
    return auto_catch_void([t, u, s, v]()
                           {
                               std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> result = torch::svd(*t);
                               *u = new torch::Tensor(std::get<0>(result));
                               *s = new torch::Tensor(std::get<1>(result));
                               *v = new torch::Tensor(std::get<2>(result)); },
                           err);
}