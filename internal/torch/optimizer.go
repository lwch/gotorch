package torch

import (
	"errors"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// #include "optimizer.h"
import "C"

type Optimizer struct {
	m    sync.Mutex
	data C.optimizer
}

func NewAdamOptimizer(params []Tensor, lr, beta1, beta2, eps, weightDecay float64) *Optimizer {
	list := make([]C.tensor, len(params))
	for i, p := range params {
		list[i] = C.tensor(p)
	}
	var err *C.char
	ptr := C.new_adam_optimizer(&err, (*C.tensor)(unsafe.Pointer(&list[0])), C.size_t(len(params)), C.double(lr), C.double(beta1), C.double(beta2), C.double(eps), C.double(weightDecay))
	if err != nil {
		panic(C.GoString(err))
	}
	optm := &Optimizer{data: ptr}
	runtime.SetFinalizer(optm, freeOptimizer)
	return optm
}

func NewAdamWOptimizer(params []Tensor, lr, beta1, beta2, eps, weightDecay float64, amsgrad bool) *Optimizer {
	list := make([]C.tensor, len(params))
	for i, p := range params {
		list[i] = C.tensor(p)
	}
	var err *C.char
	ptr := C.new_adamw_optimizer(&err, (*C.tensor)(unsafe.Pointer(&list[0])), C.size_t(len(params)), C.double(lr), C.double(beta1), C.double(beta2), C.double(eps), C.bool(amsgrad), C.double(weightDecay))
	if err != nil {
		panic(C.GoString(err))
	}
	optm := &Optimizer{data: ptr}
	runtime.SetFinalizer(optm, freeOptimizer)
	return optm
}

func freeOptimizer(optm *Optimizer) error {
	if optm == nil || optm.data == nil {
		return nil
	}
	var err *C.char
	C.free_optimizer(&err, optm.data)
	if err != nil {
		return errors.New(C.GoString(err))
	}
	optm.data = nil
	runtime.SetFinalizer(optm, nil)
	return nil
}

func (optm *Optimizer) Step() {
	optm.m.Lock()
	defer optm.m.Unlock()
	var err *C.char
	C.optimizer_step(&err, optm.data)
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

type OptimizerState struct {
	created time.Time
	data    C.optimizer_state
}

func (optm *Optimizer) GetState() *OptimizerState {
	var err *C.char
	data := C.optimizer_get_state(&err, optm.data)
	if err != nil {
		panic(C.GoString(err))
	}
	os := &OptimizerState{
		created: time.Now(),
		data:    data,
	}
	runtime.SetFinalizer(os, freeOptimizerState)
	return os
}

func freeOptimizerState(os *OptimizerState) error {
	if os == nil || os.data == nil {
		return nil
	}
	C.optimizer_state_free(os.data)
	os.data = nil
	runtime.SetFinalizer(os, nil)
	return nil
}

func (os *OptimizerState) Size() int {
	var err *C.char
	size := C.optimizer_state_count(&err, os.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return int(size)
}

func (os *OptimizerState) Get(index int) []Tensor {
	var err *C.char
	size := C.optimizer_state_size(&err, os.data, C.size_t(index))
	if err != nil {
		panic(C.GoString(err))
	}
	var tensors []Tensor
	for i := 0; i < int(size); i++ {
		var err *C.char
		tensor := C.optimizer_state_get(&err, os.data, C.size_t(index), C.size_t(i))
		if err != nil {
			panic(C.GoString(err))
		}
		tensors = append(tensors, Tensor(tensor))
	}
	return tensors
}

func (os *OptimizerState) Set(index int, values []Tensor) {
	var err *C.char
	size := C.optimizer_state_size(&err, os.data, C.size_t(index))
	if err != nil {
		panic(C.GoString(err))
	}
	if size != 0 && len(values) != int(size) {
		panic("invalid size")
	}
	for i := 0; i < int(size); i++ {
		var err *C.char
		C.optimizer_state_set(&err, os.data, C.size_t(index), C.size_t(i), C.tensor(values[i]))
		if err != nil {
			panic(C.GoString(err))
		}
	}
}
