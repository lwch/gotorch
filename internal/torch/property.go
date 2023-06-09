package torch

import "unsafe"

// #include "tensor.h"
import "C"

func (t *Tensor) ElemSize() int {
	return int(C.tensor_elem_size(t.data))
}

func (t *Tensor) ElemCount() int {
	return int(C.tensor_elem_count(t.data))
}

func (t *Tensor) Dims() int {
	return int(C.tensor_dims(t.data))
}

func (t *Tensor) Shapes() []int64 {
	size := t.Dims()
	shapes := make([]int64, size)
	C.tensor_shapes(t.data, (*C.int64_t)(unsafe.Pointer(&shapes[0])))
	return shapes
}

func (t *Tensor) Reshape(shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_reshape(t.data, shapes, size)
	return &Tensor{data: ptr}
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ptr := C.tensor_transpose(t.data, C.int64_t(dim1), C.int64_t(dim2))
	return &Tensor{data: ptr}
}
