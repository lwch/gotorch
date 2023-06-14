#ifndef __GOTORCH_TENSOR_H__
#define __GOTORCH_TENSOR_H__

#include <stdint.h>
#include <stdbool.h>
#include "api.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // init
    GOTORCH_API tensor new_tensor(char **err);
    GOTORCH_API void free_tensor(tensor t);
    GOTORCH_API tensor tensor_arange(char **err, int end, int dtype);
    GOTORCH_API tensor tensor_zeros(char **err, int64_t *shape, size_t shape_len, int dtype);
    GOTORCH_API tensor tensor_from_data(char **err, void *data, int64_t *shape, size_t shape_len, int dtype);
    GOTORCH_API void tensor_copy_data(tensor t, void *data);
    GOTORCH_API void tensor_set_requires_grad(char **err, tensor t, bool b);
    // shapes
    GOTORCH_API tensor tensor_reshape(char **err, tensor t, int64_t *shape, size_t shape_len);
    GOTORCH_API tensor tensor_transpose(char **err, tensor t, int64_t dim1, int64_t dim2);
    GOTORCH_API tensor tensor_narrow(char **err, tensor t, int64_t dim, int64_t start, int64_t length);
    GOTORCH_API tensor tensor_vstack(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_hstack(char **err, tensor a, tensor b);
    GOTORCH_API tensor tensor_view(char **err, tensor a, int64_t *shape, size_t shape_len);
    GOTORCH_API tensor tensor_permute(char **err, tensor a, int64_t *dims, size_t dims_len);
    // property
    GOTORCH_API size_t tensor_elem_size(tensor t);
    GOTORCH_API size_t tensor_elem_count(tensor t);
    GOTORCH_API int tensor_scalar_type(tensor t);
    GOTORCH_API size_t tensor_dims(tensor t);
    GOTORCH_API void tensor_shapes(tensor t, int64_t *shapes);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_TENSOR_H__