package tensor

import (
	"sync"
)

// #include "tensor.h"
import "C"

type Tensor struct {
	mFree sync.Mutex
	data  C.tensor
}

func New() *Tensor {
	ptr := C.new_tensor()
	return &Tensor{data: ptr}
}

func ARange(n int, dtype ScalarType) *Tensor {
	ptr := C.tensor_arange(C.int(n), C.int(dtype))
	return &Tensor{data: ptr}
}

func (t *Tensor) Free() {
	t.mFree.Lock()
	defer t.mFree.Unlock()
	// allready freed
	if t.data == nil {
		return
	}
	C.free_tensor(t.data)
	t.data = nil
}

func (t *Tensor) MatMul(t2 *Tensor) *Tensor {
	ptr := C.tensor_matmul(t.data, t2.data)
	return &Tensor{data: ptr}
}
