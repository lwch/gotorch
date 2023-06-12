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
    GOTORCH_API tensor new_tensor();
    GOTORCH_API void free_tensor(tensor t);
    GOTORCH_API tensor tensor_arange(int end, int dtype);
    GOTORCH_API tensor tensor_zeros(int64_t *shape, size_t shape_len, int dtype);
    GOTORCH_API tensor tensor_from_data(void *data, int64_t *shape, size_t shape_len, int dtype);
    GOTORCH_API void tensor_copy_data(tensor t, void *data);
    GOTORCH_API tensor tensor_reshape(tensor t, int64_t *shape, size_t shape_len);
    GOTORCH_API tensor tensor_transpose(tensor t, int64_t dim1, int64_t dim2);
    GOTORCH_API void tensor_set_requires_grad(tensor t, bool b);
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