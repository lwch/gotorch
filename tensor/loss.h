#ifndef __GOTORCH_LOSS_H__
#define __GOTORCH_LOSS_H__

#include <stdint.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    tensor new_mse_loss(tensor pred, tensor target, int64_t reduction);
    tensor new_cross_entropy_loss(tensor pred, tensor target,
                                  int64_t reduction,
                                  int64_t ignore_index,
                                  double label_smoothing);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_LOSS_H__