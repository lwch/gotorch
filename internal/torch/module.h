#ifndef __GOTORCH_MODULE_H__
#define __GOTORCH_MODULE_H__

#include "api.h"

#ifdef __cplusplus
extern "C"
{
#endif

    GOTORCH_API module new_linear(char **err, int64_t in_features, int64_t out_features);
    GOTORCH_API tensor linear_forward(char **err, module m, tensor x);

    GOTORCH_API module new_layer_norm(char **err, int64_t *shape, size_t shape_len);
    GOTORCH_API tensor layer_norm_forward(char **err, module m, tensor x);

    GOTORCH_API size_t module_parameters(char **err, module m, tensor *parameters);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_MODULE_H__