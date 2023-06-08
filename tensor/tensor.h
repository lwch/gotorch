#ifndef __GOTORCH_TENSOR_H__
#define __GOTORCH_TENSOR_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
    typedef torch::Tensor *tensor;
#else
typedef void *tensor;
#endif

    // init
    tensor new_tensor();
    void free_tensor(tensor t);
    tensor tensor_arange(int end, int dtype);
    tensor tensor_zeros(int64_t *shape, size_t shape_len, int dtype);
    tensor tensor_from_data(void *data, int64_t *shape, size_t shape_len, int dtype);
    void tensor_copy_data(tensor t, void *data);
    // property
    int tensor_elem_size(tensor t);
    int tensor_elem_count(tensor t);
    int tensor_scalar_type(tensor t);
    tensor tensor_reshape(tensor t, int64_t *shape, size_t shape_len);
    // operator
    tensor tensor_matmul(tensor a, tensor b);
    tensor tensor_add(tensor a, tensor b);
    tensor tensor_sub(tensor a, tensor b);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_TENSOR_H__