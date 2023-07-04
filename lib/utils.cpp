#include <torch/torch.h>
#include "exception.hpp"
#include "tensor.h"

tensor scaled_dot_product_attention(char **err,
                                    tensor q, tensor k, tensor v,
                                    tensor mask, double dropout, bool is_causal)
{
    return auto_catch_tensor([q, k, v, mask, dropout, is_causal]()
                             {
                                if (mask)
                                {
                                    return new torch::Tensor(torch::scaled_dot_product_attention(*q, *k, *v, *mask, dropout, is_causal));
                                }
                                else
                                {
                                    return new torch::Tensor(torch::scaled_dot_product_attention(*q, *k, *v, torch::nullopt, dropout, is_causal));
                                } },
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