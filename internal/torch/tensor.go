package torch

import (
	"sync"

	"github.com/lwch/gotorch/consts"
)

// #include "tensor.h"
import "C"

type Tensor struct {
	mFree sync.Mutex
	data  C.tensor
}

func ARange(n int, dtype consts.ScalarType) *Tensor {
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

func (t *Tensor) ScalarType() consts.ScalarType {
	return consts.ScalarType(C.tensor_scalar_type(t.data))
}
