#include <torch/torch.h>
#include <stdio.h>
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
    return new torch::Tensor(
        torch::from_blob(data,
                         torch::IntArrayRef(shape, shape_len),
                         torch::dtype(torch::ScalarType(dtype))));
}

void tensor_copy_data(tensor t, void *data)
{
    memcpy(data, t->data_ptr(), t->numel() * t->element_size());
}

tensor tensor_matmul(tensor a, tensor b)
{
    return new torch::Tensor(a->matmul(*b));
}

tensor tensor_add(tensor a, tensor b)
{
    return new torch::Tensor(a->add(*b));
}

tensor tensor_sub(tensor a, tensor b)
{
    return new torch::Tensor(a->sub(*b));
}