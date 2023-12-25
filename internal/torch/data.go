package torch

import (
	"fmt"
	"unsafe"

	"github.com/lwch/gotorch/consts"
)

// #include "tensor.h"
import "C"

func FromUint8(data []uint8, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cInts[uint8, C.uint8_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KUint8), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromInt8(data []int8, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cInts[int8, C.int8_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KInt8), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromInt16(data []int16, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cInts[int16, C.int16_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(&pointer), shapes, size,
		C.int8_t(consts.KInt16), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromInt32(data []int32, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cInts[int32, C.int32_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KInt32), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromInt64(data []int64, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cInts[int64, C.int64_t](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KInt64), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromFloat32(data []float32, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cFloats[float32, C.float](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KFloat), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromFloat64(data []float64, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cFloats[float64, C.double](data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KDouble), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FromBool(data []bool, shape []int64, device consts.DeviceType) Tensor {
	pointer, _ := cBool(data)
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	ptr := C.tensor_from_data(&err, unsafe.Pointer(pointer), shapes, size,
		C.int8_t(consts.KBool), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Uint8Value(t Tensor) []uint8 {
	tp := ScalarType(t)
	if tp != consts.KUint8 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.uint8_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCInts[C.uint8_t, uint8](value)
}

func Int8Value(t Tensor) []int8 {
	tp := ScalarType(t)
	if tp != consts.KInt8 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int8_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCInts[C.int8_t, int8](value)
}

func Int16Value(t Tensor) []int16 {
	tp := ScalarType(t)
	if tp != consts.KInt16 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int16_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCInts[C.int16_t, int16](value)
}

func Int32Value(t Tensor) []int32 {
	tp := ScalarType(t)
	if tp != consts.KInt32 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int32_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCInts[C.int32_t, int32](value)
}

func Int64Value(t Tensor) []int64 {
	tp := ScalarType(t)
	if tp != consts.KInt64 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.int64_t, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCInts[C.int64_t, int64](value)
}

func Float32Value(t Tensor) []float32 {
	tp := ScalarType(t)
	if tp != consts.KFloat {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.float, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCFloats[C.float, float32](value)
}

func Float64Value(t Tensor) []float64 {
	tp := ScalarType(t)
	if tp != consts.KDouble {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.double, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCFloats[C.double, float64](value)
}

func BoolValue(t Tensor) []bool {
	tp := ScalarType(t)
	if tp != consts.KBool {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]C.bool, ElemCount(t))
	C.tensor_copy_data(C.tensor(t), unsafe.Pointer(&value[0]))
	return fromCBool(value)
}
