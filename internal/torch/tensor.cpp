#include <torch/torch.h>
#include "tensor.h"

tensor new_tensor()
{
    return new torch::Tensor();
}

void free_tensor(tensor t)
{
    delete t;
}

tensor tensor_arange(int end, int dtype)
{
    return new torch::Tensor(torch::arange(end, torch::dtype(torch::ScalarType(dtype))));
}

tensor tensor_zeros(int64_t *shape, size_t shape_len, int dtype)
{
    return new torch::Tensor(
        torch::zeros(torch::IntArrayRef(shape, shape_len),
                     torch::dtype(torch::ScalarType(dtype))));
}

tensor tensor_from_data(void *data, int64_t *shape, size_t shape_len, int dtype)
{
    torch::Tensor zeros = torch::zeros(torch::IntArrayRef(shape, shape_len),
                                       torch::dtype(torch::ScalarType(dtype)));
    memcpy(zeros.data_ptr(), data, zeros.numel() * zeros.element_size());
    return new torch::Tensor(zeros);
}

void tensor_copy_data(tensor t, void *data)
{
    memcpy(data, t->data_ptr(), t->numel() * t->element_size());
}

size_t tensor_elem_size(tensor t)
{
    return t->element_size();
}

size_t tensor_elem_count(tensor t)
{
    return t->numel();
}

int tensor_scalar_type(tensor t)
{
    return int(t->scalar_type());
}

size_t tensor_dims(tensor t)
{
    return t->dim();
}

void tensor_shapes(tensor t, int64_t *shapes)
{
    size_t dim = t->dim();
    for (size_t i = 0; i < dim; i++)
    {
        shapes[i] = t->size(i);
    }
}

tensor tensor_reshape(tensor t, int64_t *shape, size_t shape_len)
{
    return new torch::Tensor(t->reshape(torch::IntArrayRef(shape, shape_len)));
}

tensor tensor_transpose(tensor t, int64_t dim1, int64_t dim2)
{
    return new torch::Tensor(t->transpose(dim1, dim2));
}

void tensor_set_requires_grad(tensor t, bool b)
{
    t->set_requires_grad(b);
}