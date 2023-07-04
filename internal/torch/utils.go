package torch

// #include <stdint.h>
// #include <stdbool.h>
// #include "tensor.h"
import "C"
import "unsafe"

func ScaledDotProductAttention(q, k, v, mask *Tensor, drouput float64, isCausal bool) *Tensor {
	var err *C.char
	var maskPtr C.tensor
	if mask != nil {
		maskPtr = mask.data
	}
	ptr := C.scaled_dot_product_attention(&err, q.data, k.data, v.data, maskPtr, C.double(drouput), C.bool(isCausal))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func ClipGradNorm(params []*Tensor, max, t float64) {
	cParams := make([]C.tensor, len(params))
	for i, p := range params {
		cParams[i] = p.data
	}
	var err *C.char
	C.clip_grad_norm(&err, (*C.tensor)(unsafe.Pointer(&cParams[0])), C.size_t(len(cParams)), C.double(max), C.double(t))
}

func fromCInts[T1 C.uint8_t | C.int8_t | C.int16_t | C.int32_t | C.int64_t,
	T2 uint8 | int8 | int16 | int32 | int64,
](arr []T1) []T2 {
	ret := make([]T2, len(arr))
	for i, v := range arr {
		ret[i] = T2(v)
	}
	return ret
}

func cInts[T1 uint8 | int8 | int16 | int32 | int64,
	T2 C.uint8_t | C.int8_t | C.int16_t | C.int32_t | C.int64_t,
](arr []T1) (*T2, C.size_t) {
	ret := make([]T2, len(arr))
	for i, v := range arr {
		ret[i] = T2(v)
	}
	return &ret[0], C.size_t(len(arr))
}

func fromCFloats[T1 C.float | C.double,
	T2 float32 | float64,
](arr []T1) []T2 {
	ret := make([]T2, len(arr))
	for i, v := range arr {
		ret[i] = T2(v)
	}
	return ret
}

func cFloats[T1 float32 | float64,
	T2 C.float | C.double,
](arr []T1) (*T2, C.size_t) {
	ret := make([]T2, len(arr))
	for i, v := range arr {
		ret[i] = T2(v)
	}
	return &ret[0], C.size_t(len(arr))
}

func fromCBool(arr []C.bool) []bool {
	ret := make([]bool, len(arr))
	for i, v := range arr {
		ret[i] = bool(v)
	}
	return ret
}

func cBool(arr []bool) (*C.bool, C.size_t) {
	ret := make([]C.bool, len(arr))
	for i, v := range arr {
		ret[i] = C.bool(v)
	}
	return &ret[0], C.size_t(len(arr))
}
