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

size_t module_parameters(char **err, module m, tensor *parameters)
{
  return auto_catch_size_t([m, parameters]()
                           { torch::autograd::variable_list list = m->parameters();
                               for (size_t i = 0; i < list.size(); i++) {
                                    parameters[i] = new torch::Tensor(list[i]);
                               }
                               return list.size(); },
                           err);
}