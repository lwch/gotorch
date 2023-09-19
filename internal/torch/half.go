package torch

import (
	"fmt"
	"unsafe"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/half"
)

// #include "tensor.h"
import "C"

func FromHalf(data []float32, shape []int64, device consts.DeviceType) *Tensor {
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
	return &Tensor{data: ptr}
}

func (t *Tensor) HalfValue() []float32 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KHalf {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint16_t, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	ret := make([]float32, len(value))
	for i, v := range value {
		ret[i] = half.DecodeHalf(uint16(v))
	}
	return ret
}

func FromBFloat16(data []float32, shape []int64, device consts.DeviceType) *Tensor {
	pointer := make([]uint16, len(data))
	for i, v := range data {
		n := *(*uint32)(unsafe.Pointer(&v))
		pointer[i] = uint16(n >> 16)
	}
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(&pointer[0]), shapes, size,
		C.int8_t(consts.KBFloat16), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) BFloat16Value() []float32 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KBFloat16 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint16_t, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	ret := make([]float32, len(value))
	for i, v := range value {
		n := uint32(v) << 16
		ret[i] = *(*float32)(unsafe.Pointer(&n))
	}
	return ret
}
