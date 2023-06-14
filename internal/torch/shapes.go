package torch

// #include "tensor.h"
import "C"

func (t *Tensor) Reshape(shape []int64) *Tensor {
	shapes, size := cInts[int64, C.int64_t](shape)
	ptr := C.tensor_reshape(t.data, shapes, size)
	return &Tensor{data: ptr}
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ptr := C.tensor_transpose(t.data, C.int64_t(dim1), C.int64_t(dim2))
	return &Tensor{data: ptr}
}

func VStack(a, b *Tensor) *Tensor {
	ptr := C.tensor_vstack(a.data, b.data)
	return &Tensor{data: ptr}
}

func HStack(a, b *Tensor) *Tensor {
	ptr := C.tensor_hstack(a.data, b.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) NArrow(dim, start, length int64) *Tensor {
	ptr := C.tensor_narrow(t.data, C.int64_t(dim), C.int64_t(start), C.int64_t(length))
	return &Tensor{data: ptr}
}

func (t *Tensor) View(shapes []int64) *Tensor {
	pointer, size := cInts[int64, C.int64_t](shapes)
	ptr := C.tensor_view(t.data, pointer, size)
	return &Tensor{data: ptr}
}

func (t *Tensor) Permute(dims []int64) *Tensor {
	pointer, size := cInts[int64, C.int64_t](dims)
	ptr := C.tensor_permute(t.data, pointer, size)
	return &Tensor{data: ptr}
}
