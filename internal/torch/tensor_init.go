package torch

// #include "tensor.h"
import "C"

func KaimingUniform(t *Tensor, a float64) {
	var err *C.char
	C.init_kaiming_uniform(&err, t.data, C.double(a))
	if err != nil {
		panic(C.GoString(err))
	}
}

func XaiverUniform(t *Tensor, gain float64) {
	var err *C.char
	C.init_xaiver_uniform(&err, t.data, C.double(gain))
	if err != nil {
		panic(C.GoString(err))
	}
}

func Normal(t *Tensor, mean, std float64) {
	var err *C.char
	C.init_normal(&err, t.data, C.double(mean), C.double(std))
	if err != nil {
		panic(C.GoString(err))
	}
}
