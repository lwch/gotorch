package torch

import (
	"github.com/lwch/gotorch/consts"
)

// #include "tensor.h"
import "C"

type Tensor C.tensor

func ARange(n int, dtype consts.ScalarType, device consts.DeviceType) Tensor {
	var err *C.char
	ptr := C.tensor_arange(&err, C.int(n), C.int8_t(dtype), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Zeros(shape []int64, dtype consts.ScalarType, device consts.DeviceType) Tensor {
	shapes := make([]C.int64_t, len(shape))
	for i, s := range shape {
		shapes[i] = C.int64_t(s)
	}
	var err *C.char
	ptr := C.tensor_zeros(&err, &shapes[0], C.size_t(len(shape)), C.int8_t(dtype), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func FreeTensor(t Tensor) {
	C.free_tensor(C.tensor(t))
}

func ScalarType(t Tensor) consts.ScalarType {
	return consts.ScalarType(C.tensor_scalar_type(C.tensor(t)))
}

func DeviceType(t Tensor) consts.DeviceType {
	return consts.DeviceType(C.tensor_device_type(C.tensor(t)))
}

func SetRequiresGrad(t Tensor, b bool) {
	var err *C.char
	C.tensor_set_requires_grad(&err, C.tensor(t), C.bool(b))
	if err != nil {
		panic(C.GoString(err))
	}
}

func ToDevice(t Tensor, device consts.DeviceType) Tensor {
	var err *C.char
	ptr := C.tensor_to_device(&err, C.tensor(t), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func ToScalarType(t Tensor, dtype consts.ScalarType) Tensor {
	var err *C.char
	ptr := C.tensor_to_scalar_type(&err, C.tensor(t), C.int8_t(dtype))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Detach(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_detach(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}
