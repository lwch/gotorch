package tensor

import (
	"fmt"
	"unsafe"
)

// #include "tensor.h"
import "C"

func FromUint8(data []uint8, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KUint8))
	return &Tensor{data: ptr}
}

func FromInt8(data []int8, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KInt8))
	return &Tensor{data: ptr}
}

func FromInt16(data []int16, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KInt16))
	return &Tensor{data: ptr}
}

func FromInt32(data []int32, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KInt32))
	return &Tensor{data: ptr}
}

func FromInt64(data []int64, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KInt64))
	return &Tensor{data: ptr}
}

func FromFloat32(data []float32, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KFloat))
	return &Tensor{data: ptr}
}

func FromFloat64(data []float64, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KDouble))
	return &Tensor{data: ptr}
}

func FromBool(data []bool, shape []int) *Tensor {
	shapes, size := cints(shape)
	ptr := C.tensor_from_data(unsafe.Pointer(&data[0]), shapes, size, C.int(KBool))
	return &Tensor{data: ptr}
}

func (t *Tensor) ScalarType() ScalarType {
	return ScalarType(C.tensor_scalar_type(t.data))
}

func (t *Tensor) Uint8Value() []uint8 {
	tp := t.ScalarType()
	if t.ScalarType() != KUint8 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]uint8, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}

func (t *Tensor) Int8Value() []int8 {
	tp := t.ScalarType()
	if t.ScalarType() != KInt8 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]int8, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}

func (t *Tensor) Int16Value() []int16 {
	tp := t.ScalarType()
	if t.ScalarType() != KInt16 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]int16, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}

func (t *Tensor) Int32Value() []int32 {
	tp := t.ScalarType()
	if t.ScalarType() != KInt32 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]int32, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}

func (t *Tensor) Int64Value() []int64 {
	tp := t.ScalarType()
	if t.ScalarType() != KInt64 {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]int64, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}

func (t *Tensor) Float32Value() []float32 {
	tp := t.ScalarType()
	if t.ScalarType() != KFloat {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]float32, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}

func (t *Tensor) Float64Value() []float64 {
	tp := t.ScalarType()
	if t.ScalarType() != KDouble {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]float64, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}

func (t *Tensor) BoolValue() []bool {
	tp := t.ScalarType()
	if t.ScalarType() != KBool {
		panic(fmt.Errorf("tensor type is %s", tp.String()))
	}
	value := make([]bool, t.ElemCount())
	C.tensor_copy_data(t.data, unsafe.Pointer(&value[0]))
	return value
}
