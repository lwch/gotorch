#ifndef __GOTORCH_OPERATOR_H__
#define __GOTORCH_OPERATOR_H__

#include <stdint.h>
#include <stdbool.h>
#include "tensor.h"

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
    GOTORCH_API tensor tensor_sigmoid(char **err, tensor a);
    GOTORCH_API tensor tensor_tanh(char **err, tensor a);
    GOTORCH_API tensor tensor_softmax(char **err, tensor a, int64_t dim);
    GOTORCH_API tensor tensor_dropout(char **err, tensor a, double p, bool train);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_OPERATOR_H__