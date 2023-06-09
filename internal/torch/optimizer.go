package torch

import (
	"sync"
	"unsafe"
)

// #include "optimizer.h"
import "C"

type Optimizer struct {
	m    sync.Mutex
	data C.optimizer
}

func NewAdamOptimizer(lr, beta1, beta2, eps, weightDecay float64) *Optimizer {
	ptr := C.new_adam_optimizer(C.double(lr), C.double(beta1), C.double(beta2), C.double(eps), C.double(weightDecay))
	return &Optimizer{data: ptr}
}

func (optm *Optimizer) Step(params []*Tensor) {
	list := make([]C.tensor, len(params))
	for i, p := range params {
		list[i] = p.data
	}
	optm.m.Lock()
	defer optm.m.Unlock()
	C.optimizer_step(optm.data, (*C.tensor)(unsafe.Pointer(&list[0])), C.size_t(len(params)))
}

func (optm *Optimizer) GetLr() float64 {
	return float64(C.optimizer_get_lr(optm.data))
}

func (optm *Optimizer) SetLr(lr float64) {
	C.optimizer_set_lr(optm.data, C.double(lr))
}
