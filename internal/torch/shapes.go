package torch

// #include "tensor.h"
import "C"

func (t *Tensor) Reshape(shape []int64) *Tensor {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_reshape(&err, t.data, shapes, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	var err *C.char
	ptr := C.tensor_transpose(&err, t.data, C.int64_t(dim1), C.int64_t(dim2))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func VStack(a, b *Tensor) *Tensor {
	var err *C.char
	ptr := C.tensor_vstack(&err, a.data, b.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func HStack(a, b *Tensor) *Tensor {
	var err *C.char
	ptr := C.tensor_hstack(&err, a.data, b.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) NArrow(dim, start, length int64) *Tensor {
	var err *C.char
	ptr := C.tensor_narrow(&err, t.data, C.int64_t(dim), C.int64_t(start), C.int64_t(length))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) View(shapes []int64) *Tensor {
	pointer, size := cInts[int64, C.int64_t](shapes)
	var err *C.char
	ptr := C.tensor_view(&err, t.data, pointer, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Permute(dims []int64) *Tensor {
	pointer, size := cInts[int64, C.int64_t](dims)
	var err *C.char
	ptr := C.tensor_permute(&err, t.data, pointer, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}
