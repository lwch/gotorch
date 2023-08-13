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
                     int64_t stride1, int64_t stride2,
                     int64_t padding1, int64_t padding2,
                     int64_t dilation, int64_t groups)
{
  return auto_catch_tensor([input, weight, bias, stride1, stride2, padding1, padding2, dilation, groups]()
                           {
                              std::vector<int64_t> stride;
                              stride.push_back(stride1);
                              stride.push_back(stride2);
                              std::vector<int64_t> padding;
                              padding.push_back(padding1);
                              padding.push_back(padding2);
                              if (bias) {
                                return new torch::Tensor(torch::conv2d(*input, *weight, *bias, stride, padding, dilation, groups));
                              } else{
                                return new torch::Tensor(torch::conv2d(*input, *weight, torch::nullopt, stride, padding, dilation, groups));
                              } },
                           err);
}

tensor tensor_conv3d(char **err, tensor input, tensor weight, tensor bias,
                     int64_t stride1, int64_t stride2, int64_t stride3,
                     int64_t padding1, int64_t padding2, int64_t padding3,
                     int64_t dilation, int64_t groups)
{
  return auto_catch_tensor([input, weight, bias, stride1, stride2, stride3, padding1, padding2, padding3, dilation, groups]()
                           {
                              std::vector<int64_t> stride;
                              stride.push_back(stride1);
                              stride.push_back(stride2);
                              stride.push_back(stride3);
                              std::vector<int64_t> padding;
                              padding.push_back(padding1);
                              padding.push_back(padding2);
                              padding.push_back(padding3);
                              if (bias) {
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