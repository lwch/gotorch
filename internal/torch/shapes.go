package torch

// #include "tensor.h"
import "C"

func Reshape(t Tensor, shape []int64) Tensor {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_reshape(&err, C.tensor(t), shapes, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Transpose(t Tensor, dim1, dim2 int64) Tensor {
	var err *C.char
	ptr := C.tensor_transpose(&err, C.tensor(t), C.int64_t(dim1), C.int64_t(dim2))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func VStack(a, b Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_vstack(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func HStack(a, b Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_hstack(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func NArrow(t Tensor, dim, start, length int64) Tensor {
	var err *C.char
	ptr := C.tensor_narrow(&err, C.tensor(t), C.int64_t(dim), C.int64_t(start), C.int64_t(length))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func View(t Tensor, shapes []int64) Tensor {
	pointer, size := cInts[int64, C.int64_t](shapes)
	var err *C.char
	ptr := C.tensor_view(&err, C.tensor(t), pointer, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Permute(t Tensor, dims []int64) Tensor {
	pointer, size := cInts[int64, C.int64_t](dims)
	var err *C.char
	ptr := C.tensor_permute(&err, C.tensor(t), pointer, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}
