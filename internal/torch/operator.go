package torch

// #include "operator.h"
import "C"

func (t *Tensor) Backward(retain bool) {
	var err *C.char
	C.tensor_backward(&err, t.data, C.bool(retain))
	if err != nil {
		panic(C.GoString(err))
	}
}

func (t *Tensor) MatMul(t2 *Tensor) *Tensor {
	var err *C.char
	ptr := C.tensor_matmul(&err, t.data, t2.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	var err *C.char
	ptr := C.tensor_add(&err, t.data, t2.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	var err *C.char
	ptr := C.tensor_sub(&err, t.data, t2.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	var err *C.char
	ptr := C.tensor_mul(&err, t.data, t2.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	var err *C.char
	ptr := C.tensor_div(&err, t.data, t2.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Pow(n float64) *Tensor {
	var err *C.char
	ptr := C.tensor_pow(&err, t.data, C.double(n))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Sqrt() *Tensor {
	var err *C.char
	ptr := C.tensor_sqrt(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Log() *Tensor {
	var err *C.char
	ptr := C.tensor_log(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Exp() *Tensor {
	var err *C.char
	ptr := C.tensor_exp(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Neg() *Tensor {
	var err *C.char
	ptr := C.tensor_neg(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Abs() *Tensor {
	var err *C.char
	ptr := C.tensor_abs(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Max(dim int64, keepdim bool) *Tensor {
	var err *C.char
	ptr := C.tensor_max(&err, t.data, C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Min(dim int64, keepdim bool) *Tensor {
	var err *C.char
	ptr := C.tensor_min(&err, t.data, C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Sum(dim int64, keepdim bool) *Tensor {
	var err *C.char
	ptr := C.tensor_sum(&err, t.data, C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Mean(dim int64, keepdim bool) *Tensor {
	var err *C.char
	ptr := C.tensor_mean(&err, t.data, C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Var(dim int64, unbiased, keepdim bool) *Tensor {
	var err *C.char
	ptr := C.tensor_var(&err, t.data, C.int64_t(dim), C.bool(unbiased), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Relu() *Tensor {
	var err *C.char
	ptr := C.tensor_relu(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Gelu(tanh bool) *Tensor {
	var err *C.char
	ptr := C.tensor_gelu(&err, t.data, C.bool(tanh))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Sigmoid() *Tensor {
	var err *C.char
	ptr := C.tensor_sigmoid(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Tanh() *Tensor {
	var err *C.char
	ptr := C.tensor_tanh(&err, t.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Softmax(dim int64) *Tensor {
	var err *C.char
	ptr := C.tensor_softmax(&err, t.data, C.int64_t(dim))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Dropout(p float64, train bool) *Tensor {
	var err *C.char
	ptr := C.tensor_dropout(&err, t.data, C.double(p), C.bool(train))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

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
