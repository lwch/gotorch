package torch

import (
	"fmt"
	"unsafe"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/half"
)

// #include "tensor.h"
import "C"

func FromHalf(data []float32, shape []int64, device consts.DeviceType) Tensor {
	pointer := make([]uint16, len(data))
	for i, v := range data {
		pointer[i] = half.EncodeHalf(v)
	}
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(&pointer[0]), shapes, size,
		C.int8_t(consts.KHalf), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromHalfRaw(data []uint16, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cInts[uint16, C.uint16_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KHalf), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func HalfValue(t Tensor) []float32 {
	tp := ScalarType(t)
	if tp != consts.KHalf {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint16_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	ret := make([]float32, len(value))
	for i, v := range value {
		ret[i] = half.DecodeHalf(uint16(v))
	}
	return ret
}

func HalfRaw(t Tensor) []uint16 {
	tp := ScalarType(t)
	if tp != consts.KHalf {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint16_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	ret := make([]uint16, len(value))
	for i, v := range value {
		ret[i] = uint16(v)
	}
	return ret
}

func FromBFloat16(data []float32, shape []int64, device consts.DeviceType) Tensor {
	pointer := make([]uint16, len(data))
	for i, v := range data {
		pointer[i] = half.EncodeBFloat16(v)
	}
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(&pointer[0]), shapes, size,
		C.int8_t(consts.KBFloat16), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromBFloat16Raw(data []uint16, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cInts[uint16, C.uint16_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KBFloat16), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func BFloat16Value(t Tensor) []float32 {
	tp := ScalarType(t)
	if tp != consts.KBFloat16 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint16_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	ret := make([]float32, len(value))
	for i, v := range value {
		ret[i] = half.DecodeBFloat16(uint16(v))
	}
	return ret
}

func BFloat16Raw(t Tensor) []uint16 {
	tp := ScalarType(t)
	if tp != consts.KBFloat16 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint16_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	ret := make([]uint16, len(value))
	for i, v := range value {
		ret[i] = uint16(v)
	}
	return ret
}
