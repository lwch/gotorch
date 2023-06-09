#ifndef __GOTORCH_OPERATOR_H__
#define __GOTORCH_OPERATOR_H__

#include <stdint.h>
#include <stdbool.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    tensor tensor_matmul(tensor a, tensor b);
    tensor tensor_add(tensor a, tensor b);
    tensor tensor_sub(tensor a, tensor b);
    tensor tensor_mul(tensor a, tensor b);
    tensor tensor_div(tensor a, tensor b);
    tensor tensor_pow(tensor a, double n);
    tensor tensor_sqrt(tensor a);
    tensor tensor_log(tensor a);
    tensor tensor_exp(tensor a);
    tensor tensor_neg(tensor a);
    tensor tensor_abs(tensor a);
    tensor tensor_max(tensor a, int64_t dim, bool keepdim);
    tensor tensor_min(tensor a, int64_t dim, bool keepdim);
    tensor tensor_sum(tensor a, int64_t dim, bool keepdim);
    tensor tensor_mean(tensor a, int64_t dim, bool keepdim);
    tensor tensor_relu(tensor a);
    tensor tensor_sigmoid(tensor a);
    tensor tensor_tanh(tensor a);
    tensor tensor_softmax(tensor a, int64_t dim);
    tensor tensor_dropout(tensor a, double p, bool train);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_OPERATOR_H__