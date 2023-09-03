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

    GOTORCH_API module new_attention(char **err, int64_t embed_dim, int64_t num_heads, double dropout);
    GOTORCH_API tensor attention_forward(char **err, module m, tensor q, tensor k, tensor v, tensor mask, tensor *score);

    GOTORCH_API module new_embedding(char **err, int64_t num_embeddings, int64_t embedding_dim, int64_t padding_idx);
    GOTORCH_API tensor embedding_forward(char **err, module m, tensor x);

    GOTORCH_API void module_to_device(char **err, module m, int8_t device);
    GOTORCH_API void module_to_scalar_type(char **err, module m, int8_t type);
    GOTORCH_API size_t module_parameters(char **err, module m, tensor *parameters);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_MODULE_H__