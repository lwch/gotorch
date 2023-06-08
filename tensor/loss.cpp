#include <torch/torch.h>
#include "tensor.h"

tensor tensor_mse_loss(tensor pred, tensor target, int64_t reduction)
{
    return new torch::Tensor(torch::mse_loss(*pred, *target, reduction));
}