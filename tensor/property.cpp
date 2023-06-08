#include <torch/torch.h>
#include "tensor.h"

int tensor_elem_size(tensor t)
{
    return t->element_size();
}

int tensor_elem_count(tensor t)
{
    return t->numel();
}

int tensor_scalar_type(tensor t)
{
    return int(t->scalar_type());
}

tensor tensor_reshape(tensor t, int64_t *shape, size_t shape_len)
{
    return new torch::Tensor(t->reshape(torch::IntArrayRef(shape, shape_len)));
}