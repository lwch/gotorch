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

func ARange(n int, dtype consts.ScalarType, device consts.DeviceType) *Tensor {
	var err *C.char
	ptr := C.tensor_arange(&err, C.int(n), C.int8_t(dtype), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func Zeros(shape []int64, dtype consts.ScalarType, device consts.DeviceType) *Tensor {
	shapes := make([]C.int64_t, len(shape))
	for i, s := range shape {
		shapes[i] = C.int64_t(s)
	}
	var err *C.char
	ptr := C.tensor_zeros(&err, &shapes[0], C.size_t(len(shape)), C.int8_t(dtype), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
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

func (t *Tensor) SetRequiresGrad(b bool) {
	var err *C.char
	C.tensor_set_requires_grad(&err, t.data, C.bool(b))
	if err != nil {
		panic(C.GoString(err))
	}
}

func (t *Tensor) ToDevice(device consts.DeviceType) *Tensor {
	var err *C.char
	ptr := C.tensor_to_device(&err, t.data, C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}
