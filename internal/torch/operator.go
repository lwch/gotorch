package torch

// #include "operator.h"
import "C"

func (t *Tensor) MatMul(t2 *Tensor) *Tensor {
	ptr := C.tensor_matmul(t.data, t2.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	ptr := C.tensor_add(t.data, t2.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	ptr := C.tensor_sub(t.data, t2.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	ptr := C.tensor_mul(t.data, t2.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	ptr := C.tensor_div(t.data, t2.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Pow(n float64) *Tensor {
	ptr := C.tensor_pow(t.data, C.double(n))
	return &Tensor{data: ptr}
}

func (t *Tensor) Sqrt() *Tensor {
	ptr := C.tensor_sqrt(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Log() *Tensor {
	ptr := C.tensor_log(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Exp() *Tensor {
	ptr := C.tensor_exp(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Neg() *Tensor {
	ptr := C.tensor_neg(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Abs() *Tensor {
	ptr := C.tensor_abs(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Max(dim int64, keepdim bool) *Tensor {
	ptr := C.tensor_max(t.data, C.int64_t(dim), C.bool(keepdim))
	return &Tensor{data: ptr}
}

func (t *Tensor) Min(dim int64, keepdim bool) *Tensor {
	ptr := C.tensor_min(t.data, C.int64_t(dim), C.bool(keepdim))
	return &Tensor{data: ptr}
}

func (t *Tensor) Sum(dim int64, keepdim bool) *Tensor {
	ptr := C.tensor_sum(t.data, C.int64_t(dim), C.bool(keepdim))
	return &Tensor{data: ptr}
}

func (t *Tensor) Mean(dim int64, keepdim bool) *Tensor {
	ptr := C.tensor_mean(t.data, C.int64_t(dim), C.bool(keepdim))
	return &Tensor{data: ptr}
}

func (t *Tensor) Relu() *Tensor {
	ptr := C.tensor_relu(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Sigmoid() *Tensor {
	ptr := C.tensor_sigmoid(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Tanh() *Tensor {
	ptr := C.tensor_tanh(t.data)
	return &Tensor{data: ptr}
}

func (t *Tensor) Softmax(dim int64) *Tensor {
	ptr := C.tensor_softmax(t.data, C.int64_t(dim))
	return &Tensor{data: ptr}
}

func (t *Tensor) Dropout(p float64, train bool) *Tensor {
	ptr := C.tensor_dropout(t.data, C.double(p), C.bool(train))
	return &Tensor{data: ptr}
}
