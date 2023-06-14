package torch

import (
	"fmt"
	"unsafe"

	"github.com/lwch/gotorch/consts"
)

// #include "tensor.h"
import "C"

func FromUint8(data []uint8, shape []int64) *Tensor {
	pointer, _ := cInts[uint8, C.uint8_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size, C.int(consts.KUint8))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func FromInt8(data []int8, shape []int64) *Tensor {
	pointer, _ := cInts[int8, C.int8_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size, C.int(consts.KInt8))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func FromInt16(data []int16, shape []int64) *Tensor {
	pointer, _ := cInts[int16, C.int16_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(&pointer), shapes, size, C.int(consts.KInt16))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func FromInt32(data []int32, shape []int64) *Tensor {
	pointer, _ := cInts[int32, C.int32_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size, C.int(consts.KInt32))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func FromInt64(data []int64, shape []int64) *Tensor {
	pointer, _ := cInts[int64, C.int64_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size, C.int(consts.KInt64))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func FromFloat32(data []float32, shape []int64) *Tensor {
	pointer, _ := cFloats[float32, C.float](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size, C.int(consts.KFloat))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func FromFloat64(data []float64, shape []int64) *Tensor {
	pointer, _ := cFloats[float64, C.double](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size, C.int(consts.KDouble))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func FromBool(data []bool, shape []int64) *Tensor {
	pointer, _ := cBool(data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size, C.int(consts.KBool))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Uint8Value() []uint8 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KUint8 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint8_t, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCInts[C.uint8_t, uint8](value)
}

func (t *Tensor) Int8Value() []int8 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KInt8 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int8_t, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCInts[C.int8_t, int8](value)
}

func (t *Tensor) Int16Value() []int16 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KInt16 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int16_t, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCInts[C.int16_t, int16](value)
}

func (t *Tensor) Int32Value() []int32 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KInt32 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int32_t, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCInts[C.int32_t, int32](value)
}

func (t *Tensor) Int64Value() []int64 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KInt64 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int64_t, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCInts[C.int64_t, int64](value)
}

func (t *Tensor) Float32Value() []float32 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KFloat {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.float, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCFloats[C.float, float32](value)
}

func (t *Tensor) Float64Value() []float64 {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KDouble {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.double, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCFloats[C.double, float64](value)
}

func (t *Tensor) BoolValue() []bool {
	tp := t.ScalarType()
	if t.ScalarType() != consts.KBool {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.bool, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return fromCBool(value)
}
