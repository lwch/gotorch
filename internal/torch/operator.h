#ifndef __GOTORCH_OPERATOR_H__
#define __GOTORCH_OPERATOR_H__

#include "tensor.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    GOTORCH_API void tensor_backward(char **err, tensor a, bool retain);
    GOTORCH_API tensor tensor_matmul(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_add(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_sub(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_mul(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_div(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_pow(char **err, tensor a, double n);
    GOTORCH_API tensor tensor_sqrt(char **err, tensor a);
    GOTORCH_API tensor tensor_rsqrt(char **err, tensor a);
    GOTORCH_API tensor tensor_log(char **err, tensor a);
    GOTORCH_API tensor tensor_exp(char **err, tensor a);
    GOTORCH_API tensor tensor_neg(char **err, tensor a);
    GOTORCH_API tensor tensor_abs(char **err, tensor a);
    GOTORCH_API tensor tensor_max(char **err, tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_min(char **err, tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_sum(char **err, tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_mean(char **err, tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_var(char **err, tensor a, int64_t dim, bool unbiased, bool keepdim);
    GOTORCH_API tensor tensor_relu(char **err, tensor a);
    GOTORCH_API tensor tensor_gelu(char **err, tensor a, bool tanh);
    GOTORCH_API tensor tensor_leaky_relu(char **err, tensor a, double negative_slope);
    GOTORCH_API tensor tensor_silu(char **err, tensor a);
    GOTORCH_API tensor tensor_sigmoid(char **err, tensor a);
    GOTORCH_API tensor tensor_tanh(char **err, tensor a);
    GOTORCH_API tensor tensor_softmax(char **err, tensor a, int64_t dim);
    GOTORCH_API tensor tensor_softmax1(char **err, tensor a, int64_t dim);
    GOTORCH_API tensor tensor_dropout(char **err, tensor a, double p, bool train);
    GOTORCH_API tensor tensor_unsqueeze(char **err, tensor a, int64_t dim);
    GOTORCH_API tensor tensor_squeeze(char **err, tensor a, int64_t dim);
    GOTORCH_API tensor tensor_contiguous(char **err, tensor a);
    GOTORCH_API tensor tensor_expand(char **err, tensor a, int64_t *sizes, size_t len);
    GOTORCH_API tensor tensor_gather(char **err, tensor a, int64_t dim, tensor index);
    GOTORCH_API tensor tensor_clamp(char **err, tensor a, double min, double max);
    GOTORCH_API tensor tensor_min_tensor(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_max_tensor(char **err, tensor a, tensor b);
    // conv
    GOTORCH_API tensor tensor_conv1d(char **err, tensor input, tensor weight, tensor bias, int64_t stride,
                                     int64_t padding, int64_t dilation, int64_t groups);
    GOTORCH_API tensor tensor_conv2d(char **err, tensor input, tensor weight, tensor bias, int64_t stride1,
                                     int64_t stride2, int64_t padding1, int64_t padding2, int64_t dilation,
                                     int64_t groups);
    GOTORCH_API tensor tensor_conv3d(char **err, tensor input, tensor weight, tensor bias, int64_t stride1,
                                     int64_t stride2, int64_t stride3, int64_t padding1, int64_t padding2,
                                     int64_t padding3, int64_t dilation, int64_t groups);
    // transpose conv
    GOTORCH_API tensor tensor_transpose_conv1d(char **err, tensor input, tensor weight, tensor bias, int64_t stride,
                                               int64_t padding, int64_t output_padding, int64_t dilation,
                                               int64_t groups);
    GOTORCH_API tensor tensor_transpose_conv2d(char **err, tensor input, tensor weight, tensor bias, int64_t stride1,
                                               int64_t stride2, int64_t padding1, int64_t padding2,
                                               int64_t output_padding1, int64_t output_padding2, int64_t dilation,
                                               int64_t groups);
    GOTORCH_API tensor tensor_transpose_conv3d(char **err, tensor input, tensor weight, tensor bias, int64_t stride1,
                                               int64_t stride2, int64_t stride3, int64_t padding1, int64_t padding2,
                                               int64_t padding3, int64_t output_padding1, int64_t output_padding2,
                                               int64_t output_padding3, int64_t dilation, int64_t groups);
    // pool
    GOTORCH_API tensor tensor_max_pool1d(char **err, tensor self, int64_t kernel_size, int64_t stride, int64_t padding,
                                         int64_t dilation, bool ceil_mode);
    GOTORCH_API tensor tensor_max_pool2d(char **err, tensor self, int64_t kernel_size, int64_t stride, int64_t padding,
                                         int64_t dilation, bool ceil_mode);
    GOTORCH_API tensor tensor_max_pool3d(char **err, tensor self, int64_t kernel_size, int64_t stride, int64_t padding,
                                         int64_t dilation, bool ceil_mode);
    GOTORCH_API tensor tensor_avg_pool1d(char **err, tensor self, int64_t kernel_size, int64_t stride, int64_t padding,
                                         int64_t dilation, bool ceil_mode);
    GOTORCH_API tensor tensor_avg_pool2d(char **err, tensor self, int64_t kernel_size, int64_t stride, int64_t padding,
                                         int64_t dilation, bool ceil_mode);
    GOTORCH_API tensor tensor_avg_pool3d(char **err, tensor self, int64_t kernel_size, int64_t stride, int64_t padding,
                                         int64_t dilation, bool ceil_mode);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_OPERATOR_H__