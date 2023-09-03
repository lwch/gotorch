#include <torch/torch.h>
#include "exception.hpp"
#include "module.h"

module new_linear(char **err, int64_t in_features, int64_t out_features)
{
  return auto_catch_module([in_features, out_features]()
                           { return new torch::nn::LinearImpl(in_features, out_features); },
                           err);
}

tensor linear_forward(char **err, module m, tensor x)
{
  return auto_catch_tensor([m, x]()
                           {
                            torch::nn::LinearImpl* mm = dynamic_cast<torch::nn::LinearImpl*>(m);
                            return new torch::Tensor(mm->forward(*x)); },
                           err);
}

module new_layer_norm(char **err, int64_t *shape, size_t shape_len)
{
  return auto_catch_module([shape, shape_len]()
                           { std::vector<int64_t> shapes;
                             for (size_t i = 0; i < shape_len; i++) {
                                shapes.push_back(shape[i]);
                             }
                             return new torch::nn::LayerNormImpl(shapes); },
                           err);
}

tensor layer_norm_forward(char **err, module m, tensor x)
{
  return auto_catch_tensor([m, x]()
                           {
                            torch::nn::LayerNormImpl* mm = dynamic_cast<torch::nn::LayerNormImpl*>(m);
                            return new torch::Tensor(mm->forward(*x)); },
                           err);
}

module new_attention(char **err, int64_t embed_dim, int64_t num_heads, double dropout)
{
  return auto_catch_module([embed_dim, num_heads, dropout]()
                           { return new torch::nn::MultiheadAttentionImpl(
                                 torch::nn::MultiheadAttentionOptions(embed_dim, num_heads)
                                     .dropout(dropout)); },
                           err);
}

tensor attention_forward(char **err, module m, tensor q, tensor k, tensor v, tensor mask, tensor *score)
{
  return auto_catch_tensor([m, q, k, v, mask, score]()
                           { torch::nn::MultiheadAttentionImpl *mm = dynamic_cast<torch::nn::MultiheadAttentionImpl *>(m);
                             torch::Tensor _mask;
                             if (mask)
                                _mask = *mask;
                             std::tuple<torch::Tensor, torch::Tensor> result = mm->forward(*q, *k, *v, {}, true, _mask, false);
                             *score = new torch::Tensor(std::get<1>(result));
                             return new torch::Tensor(std::get<0>(result)); },
                           err);
}

void module_to_device(char **err, module m, int8_t device)
{
  auto_catch_void([m, device]()
                  { m->to(torch::DeviceType(device)); },
                  err);
}

void module_to_scalar_type(char **err, module m, int8_t type)
{
  auto_catch_void([m, type]()
                  { m->to(torch::Dtype(type)); },
                  err);
}

size_t module_parameters(char **err, module m, tensor *parameters)
{
  return auto_catch_size_t([m, parameters]()
                           {
                              torch::autograd::variable_list list = m->parameters();
                              for (size_t i = 0; i < list.size(); i++) {
                                parameters[i] = new torch::Tensor(list[i]);
                              }
                              return list.size(); },
                           err);
}