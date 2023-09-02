package torch

// #include <stdint.h>
// #include "module.h"
import "C"

type Module interface {
	Parameters() []*Tensor
	Forward(*Tensor) *Tensor
}

type Linear struct {
	m C.module
}

func NewLinear(inFeatures, outFeatures int64) *Linear {
	var err *C.char
	l := C.new_linear(&err, C.int64_t(inFeatures), C.int64_t(outFeatures))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Linear{
		m: l,
	}
}

func (l *Linear) Parameters() []*Tensor {
	params := make([]C.tensor, 2)
	var err *C.char
	C.module_parameters(&err, l.m, (*C.tensor)(&params[0]))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := make([]*Tensor, 2)
	ret[0] = &Tensor{data: params[0]}
	ret[1] = &Tensor{data: params[1]}
	return ret
}

func (l *Linear) Forward(x *Tensor) *Tensor {
	var err *C.char
	ptr := C.linear_forward(&err, l.m, x.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}
