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
	var err *C.char
	ptr := C.new_adam_optimizer(&err, C.double(lr), C.double(beta1), C.double(beta2), C.double(eps), C.double(weightDecay))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Optimizer{data: ptr}
}

func (optm *Optimizer) Step(params []*Tensor) {
	list := make([]C.tensor, len(params))
	for i, p := range params {
		list[i] = p.data
	}
	optm.m.Lock()
	defer optm.m.Unlock()
	var err *C.char
	C.optimizer_step(&err, optm.data, (*C.tensor)(unsafe.Pointer(&list[0])), C.size_t(len(params)))
	if err != nil {
		panic(C.GoString(err))
	}
}

func (optm *Optimizer) GetLr() float64 {
	var err *C.char
	lr := C.optimizer_get_lr(&err, optm.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return float64(lr)
}

func (optm *Optimizer) SetLr(lr float64) {
	var err *C.char
	C.optimizer_set_lr(&err, optm.data, C.double(lr))
	if err != nil {
		panic(C.GoString(err))
	}
}
