#include <torch/torch.h>
#include "exception.hpp"
#include "tensor.h"

tensor scaled_dot_product_attention(char **err,
                                    tensor q, tensor k, tensor v,
                                    tensor mask, double dropout, bool is_causal,
                                    tensor *score)
{
    return auto_catch_tensor([q, k, v, mask, dropout, is_causal, score]()
                             {
                                std::tuple<torch::Tensor, torch::Tensor> result;
                                if (mask)
                                {
                                    result = torch::_scaled_dot_product_attention(*q, *k, *v, *mask, dropout, true, is_causal);
                                }
                                else
                                {
                                    result = torch::_scaled_dot_product_attention(*q, *k, *v, torch::nullopt, dropout, true, is_causal);
                                }
                                *score = new torch::Tensor(std::get<1>(result));
                                return new torch::Tensor(std::get<0>(result)); },
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