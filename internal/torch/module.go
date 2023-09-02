package torch

// #include <stdint.h>
// #include "module.h"
import "C"
import (
	"github.com/lwch/gotorch/consts"
)

type Module interface {
	Parameters() []*Tensor
	Forward(*Tensor) *Tensor
	ToDevice(consts.DeviceType)
	ToScalarType(consts.ScalarType)
}

type module struct {
	m C.module
}

func (m *module) parameters(count int) []*Tensor {
	params := make([]C.tensor, count)
	var err *C.char
	C.module_parameters(&err, m.m, (*C.tensor)(&params[0]))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := make([]*Tensor, count)
	for i := 0; i < count; i++ {
		ret[i] = &Tensor{data: params[i]}
	}
	return ret
}

func (m *module) ToDevice(device consts.DeviceType) {
	var err *C.char
	C.module_to_device(&err, m.m, C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
}

func (m *module) ToScalarType(t consts.ScalarType) {
	var err *C.char
	C.module_to_scalar_type(&err, m.m, C.int8_t(t))
	if err != nil {
		panic(C.GoString(err))
	}
}

type Linear struct {
	module
}

func NewLinear(inFeatures, outFeatures int64) *Linear {
	var err *C.char
	l := C.new_linear(&err, C.int64_t(inFeatures), C.int64_t(outFeatures))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Linear{module{l}}
}

func (l *module) Forward(x *Tensor) *Tensor {
	var err *C.char
	ptr := C.linear_forward(&err, l.m, x.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (l *Linear) Parameters() []*Tensor {
	return l.parameters(2)
}

type LayerNorm struct {
	module
}

func NewLayerNorm(shape []int64) *LayerNorm {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	l := C.new_layer_norm(&err, shapes, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return &LayerNorm{module{l}}
}

func (l *LayerNorm) Forward(x *Tensor) *Tensor {
	var err *C.char
	ptr := C.layer_norm_forward(&err, l.m, x.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (l *LayerNorm) Parameters() []*Tensor {
	return l.parameters(2)
}
