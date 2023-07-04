#include <torch/torch.h>
#include "exception.hpp"
#include "operator.h"

tensor tensor_conv1d(char **err, tensor input, tensor weight, tensor bias,
                     int64_t stride, int64_t padding, int64_t dilation, int64_t groups)
{
  return auto_catch_tensor([input, weight, bias, stride, padding, dilation, groups]()
                           { if (bias) {
                                return new torch::Tensor(torch::conv1d(*input, *weight, *bias, stride, padding, dilation, groups));
                              } else{
                                return new torch::Tensor(torch::conv1d(*input, *weight, torch::nullopt, stride, padding, dilation, groups));
                              } },
                           err);
}

tensor tensor_conv2d(char **err, tensor input, tensor weight, tensor bias,
                     int64_t stride, int64_t padding, int64_t dilation, int64_t groups)
{
  return auto_catch_tensor([input, weight, bias, stride, padding, dilation, groups]()
                           { if (bias) {
                                return new torch::Tensor(torch::conv2d(*input, *weight, *bias, stride, padding, dilation, groups));
                              } else{
                                return new torch::Tensor(torch::conv2d(*input, *weight, torch::nullopt, stride, padding, dilation, groups));
                              } },
                           err);
}

tensor tensor_conv3d(char **err, tensor input, tensor weight, tensor bias,
                     int64_t stride, int64_t padding, int64_t dilation, int64_t groups)
{
  return auto_catch_tensor([input, weight, bias, stride, padding, dilation, groups]()
                           { if (bias) {
                                return new torch::Tensor(torch::conv3d(*input, *weight, *bias, stride, padding, dilation, groups));
                              } else{
                                return new torch::Tensor(torch::conv3d(*input, *weight, torch::nullopt, stride, padding, dilation, groups));
                              } },
                           err);
}

tensor tensor_max_pool1d(char **err, tensor self,
                         int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode)
{
  return auto_catch_tensor([self, kernel_size, stride, padding, dilation, ceil_mode]()
                           { return new torch::Tensor(torch::max_pool1d(*self, kernel_size, stride, padding, dilation, ceil_mode)); },
                           err);
}

tensor tensor_max_pool2d(char **err, tensor self,
                         int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode)
{
  return auto_catch_tensor([self, kernel_size, stride, padding, dilation, ceil_mode]()
                           { return new torch::Tensor(torch::max_pool2d(*self, kernel_size, stride, padding, dilation, ceil_mode)); },
                           err);
}

tensor tensor_max_pool3d(char **err, tensor self,
                         int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode)
{
  return auto_catch_tensor([self, kernel_size, stride, padding, dilation, ceil_mode]()
                           { return new torch::Tensor(torch::max_pool3d(*self, kernel_size, stride, padding, dilation, ceil_mode)); },
                           err);
}

tensor tensor_avg_pool1d(char **err, tensor self,
                         int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode)
{
  return auto_catch_tensor([self, kernel_size, stride, padding, dilation, ceil_mode]()
                           { return new torch::Tensor(torch::avg_pool1d(*self, kernel_size, stride, padding, dilation, ceil_mode)); },
                           err);
}

tensor tensor_avg_pool2d(char **err, tensor self,
                         int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode)
{
  return auto_catch_tensor([self, kernel_size, stride, padding, dilation, ceil_mode]()
                           { return new torch::Tensor(torch::avg_pool2d(*self, kernel_size, stride, padding, dilation, ceil_mode)); },
                           err);
}

tensor tensor_avg_pool3d(char **err, tensor self,
                         int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode)
{
  return auto_catch_tensor([self, kernel_size, stride, padding, dilation, ceil_mode]()
                           { return new torch::Tensor(torch::avg_pool3d(*self, kernel_size, stride, padding, dilation, ceil_mode)); },
                           err);
}