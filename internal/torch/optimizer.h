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

    optimizer new_adam_optimizer(double lr, double beta1, double beta2, double weight_decay);
    void optimizer_step(optimizer optm, tensor *params, size_t params_count);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_OPTIMIZER_H__