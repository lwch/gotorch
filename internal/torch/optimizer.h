#ifndef __GOTORCH_OPTIMIZER_H__
#define __GOTORCH_OPTIMIZER_H__

#include <stdint.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
    typedef torch::optim::Optimizer *optimizer;
#else
typedef void *optimizer;
#endif

    GOTORCH_API optimizer new_adam_optimizer(char **err, double lr, double beta1, double beta2, double eps, double weight_decay);
    GOTORCH_API optimizer new_adamw_optimizer(char **err, double lr, double beta1, double beta2, double eps, bool amsgrad, double weight_decay);
    GOTORCH_API void optimizer_step(char **err, optimizer optm, tensor *params, size_t params_count);
    GOTORCH_API double optimizer_get_lr(char **err, optimizer optm);
    GOTORCH_API void optimizer_set_lr(char **err, optimizer optm, double lr);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_OPTIMIZER_H__