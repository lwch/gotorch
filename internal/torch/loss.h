#ifndef __GOTORCH_LOSS_H__
#define __GOTORCH_LOSS_H__

#include <stdint.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    GOTORCH_API tensor new_mse_loss(char **err, tensor pred, tensor target, int64_t reduction);
    GOTORCH_API tensor new_cross_entropy_loss(char **err, tensor pred, tensor target,
                                              tensor weight,
                                              int64_t reduction,
                                              int64_t ignore_index,
                                              double label_smoothing);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_LOSS_H__