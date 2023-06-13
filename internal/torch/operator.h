#ifndef __GOTORCH_OPERATOR_H__
#define __GOTORCH_OPERATOR_H__

#include <stdint.h>
#include <stdbool.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    GOTORCH_API void tensor_backward(tensor a, bool retain);
    GOTORCH_API tensor tensor_matmul(tensor a, tensor b);
    GOTORCH_API tensor tensor_add(tensor a, tensor b);
    GOTORCH_API tensor tensor_sub(tensor a, tensor b);
    GOTORCH_API tensor tensor_mul(tensor a, tensor b);
    GOTORCH_API tensor tensor_div(tensor a, tensor b);
    GOTORCH_API tensor tensor_pow(tensor a, double n);
    GOTORCH_API tensor tensor_sqrt(tensor a);
    GOTORCH_API tensor tensor_log(tensor a);
    GOTORCH_API tensor tensor_exp(tensor a);
    GOTORCH_API tensor tensor_neg(tensor a);
    GOTORCH_API tensor tensor_abs(tensor a);
    GOTORCH_API tensor tensor_max(tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_min(tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_sum(tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_mean(tensor a, int64_t dim, bool keepdim);
    GOTORCH_API tensor tensor_var(tensor a, int64_t dim, bool unbiased, bool keepdim);
    GOTORCH_API tensor tensor_relu(tensor a);
    GOTORCH_API tensor tensor_sigmoid(tensor a);
    GOTORCH_API tensor tensor_tanh(tensor a);
    GOTORCH_API tensor tensor_softmax(tensor a, int64_t dim);
    GOTORCH_API tensor tensor_dropout(tensor a, double p, bool train);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_OPERATOR_H__