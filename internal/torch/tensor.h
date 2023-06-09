#ifndef __GOTORCH_TENSOR_H__
#define __GOTORCH_TENSOR_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
    typedef torch::Tensor *tensor;
    typedef torch::optim::Optimizer *optimizer;
#else
typedef void *tensor;
typedef void *optimizer;
#endif

    // init
    tensor new_tensor();
    void free_tensor(tensor t);
    tensor tensor_arange(int end, int dtype);
    tensor tensor_zeros(int64_t *shape, size_t shape_len, int dtype);
    tensor tensor_from_data(void *data, int64_t *shape, size_t shape_len, int dtype);
    void tensor_copy_data(tensor t, void *data);
    tensor tensor_reshape(tensor t, int64_t *shape, size_t shape_len);
    tensor tensor_transpose(tensor t, int64_t dim1, int64_t dim2);
    // property
    int tensor_elem_size(tensor t);
    int tensor_elem_count(tensor t);
    int tensor_scalar_type(tensor t);
    size_t tensor_dims(tensor t);
    void tensor_shapes(tensor t, int64_t *shapes);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_TENSOR_H__