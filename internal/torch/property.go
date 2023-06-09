package torch

import "unsafe"

// #include "tensor.h"
import "C"

func (t *Tensor) ElemSize() int64 {
	return int64(C.tensor_elem_size(t.data))
}

func (t *Tensor) ElemCount() int64 {
	return int64(C.tensor_elem_count(t.data))
}

func (t *Tensor) Dims() int64 {
	return int64(C.tensor_dims(t.data))
}

func (t *Tensor) Shapes() []int64 {
	size := t.Dims()
	shapes := make([]C.int64_t, size)
	C.tensor_shapes(t.data, (*C.int64_t)(unsafe.Pointer(&shapes[0])))
	return fromCInts[C.int64_t, int64](shapes)
}

func (t *Tensor) Reshape(shape []int64) *Tensor {
	shapes, size := cInts[int64, C.int64_t](shape)
	ptr := C.tensor_reshape(t.data, shapes, size)
	return &Tensor{data: ptr}
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ptr := C.tensor_transpose(t.data, C.int64_t(dim1), C.int64_t(dim2))
	return &Tensor{data: ptr}
}
