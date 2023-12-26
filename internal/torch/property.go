package torch

import "unsafe"

// #include "tensor.h"
import "C"

func ElemSize(t Tensor) int64 {
	return int64(C.tensor_elem_size(C.tensor(t)))
}

func ElemCount(t Tensor) int64 {
	return int64(C.tensor_elem_count(C.tensor(t)))
}

func Dims(t Tensor) int64 {
	return int64(C.tensor_dims(C.tensor(t)))
}

func Shapes(t Tensor) []int64 {
	size := Dims(t)
	shapes := make([]C.int64_t, size)
	C.tensor_shapes(C.tensor(t), (*C.int64_t)(unsafe.Pointer(&shapes[0])))
	return fromCInts[C.int64_t, int64](shapes)
}
