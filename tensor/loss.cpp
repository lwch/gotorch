#include <torch/torch.h>
#include "loss.h"

tensor new_mse_loss(tensor pred, tensor target, int64_t reduction)
{
    return new torch::Tensor(torch::mse_loss(*pred, *target, reduction));
}

tensor new_cross_entropy_loss(tensor pred, tensor target,
                              int64_t reduction,
                              int64_t ignore_index,
                              double label_smoothing)
{
    return new torch::Tensor(torch::cross_entropy_loss(*pred, *target, c10::nullopt, reduction, ignore_index, label_smoothing));
}