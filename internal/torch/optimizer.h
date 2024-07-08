#ifndef __GOTORCH_OPTIMIZER_H__
#define __GOTORCH_OPTIMIZER_H__

#include <stdint.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

    GOTORCH_API optimizer new_adam_optimizer(char **err, tensor *params, size_t params_count, double lr, double beta1, double beta2, double eps, double weight_decay);
    GOTORCH_API optimizer new_adamw_optimizer(char **err, tensor *params, size_t params_count, double lr, double beta1, double beta2, double eps, bool amsgrad, double weight_decay);
    GOTORCH_API void optimizer_step(char **err, optimizer optm);
    GOTORCH_API double optimizer_get_lr(char **err, optimizer optm);
    GOTORCH_API void optimizer_set_lr(char **err, optimizer optm, double lr);
    GOTORCH_API optimizer_state optimizer_get_state(char **err, optimizer optm);
    GOTORCH_API void optimizer_state_free(optimizer_state state);
    GOTORCH_API size_t optimizer_state_count(char **err, optimizer_state state);
    GOTORCH_API size_t optimizer_state_size(char **err, optimizer_state state, size_t index);
    GOTORCH_API tensor optimizer_state_get(char **err, optimizer_state state, size_t index, size_t key);

#ifdef __cplusplus
}
#endif

#endif // __GOTORCH_OPTIMIZER_H__