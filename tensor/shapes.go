package tensor

// #cgo CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -ltorch -lc10 -ltorch_cpu
// #include "tensor.h"
import "C"

func (t *Tensor) ElemSize() int {
	return int(C.tensor_elem_size(t.data))
}

func (t *Tensor) ElemCount() int {
	return int(C.tensor_elem_count(t.data))
}

func (t *Tensor) Reshape(shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_reshape(t.data, shapes, size)
	return &Tensor{data: ptr}
}
