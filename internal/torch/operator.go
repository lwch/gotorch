package torch

// #include "operator.h"
import "C"

func Backward(t Tensor, retain bool) {
	var err *C.char
	C.tensor_backward(&err, C.tensor(t), C.bool(retain))
	if err != nil {
		panic(C.GoString(err))
	}
}

func MatMul(t, t2 Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_matmul(&err, C.tensor(t), C.tensor(t2))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Add(t, t2 Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_add(&err, C.tensor(t), C.tensor(t2))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Sub(t, t2 Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_sub(&err, C.tensor(t), C.tensor(t2))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Mul(t, t2 Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_mul(&err, C.tensor(t), C.tensor(t2))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Div(t, t2 Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_div(&err, C.tensor(t), C.tensor(t2))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Pow(t Tensor, n float64) Tensor {
	var err *C.char
	ptr := C.tensor_pow(&err, C.tensor(t), C.double(n))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Sqrt(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_sqrt(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func RSqrt(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_rsqrt(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Log(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_log(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Exp(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_exp(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Neg(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_neg(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Abs(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_abs(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Max(t Tensor, dim int64, keepdim bool) Tensor {
	var err *C.char
	ptr := C.tensor_max(&err, C.tensor(t), C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Min(t Tensor, dim int64, keepdim bool) Tensor {
	var err *C.char
	ptr := C.tensor_min(&err, C.tensor(t), C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Sum(t Tensor, dim int64, keepdim bool) Tensor {
	var err *C.char
	ptr := C.tensor_sum(&err, C.tensor(t), C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Mean(t Tensor, dim int64, keepdim bool) Tensor {
	var err *C.char
	ptr := C.tensor_mean(&err, C.tensor(t), C.int64_t(dim), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Var(t Tensor, dim int64, unbiased, keepdim bool) Tensor {
	var err *C.char
	ptr := C.tensor_var(&err, C.tensor(t), C.int64_t(dim), C.bool(unbiased), C.bool(keepdim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Relu(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_relu(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Gelu(t Tensor, tanh bool) Tensor {
	var err *C.char
	ptr := C.tensor_gelu(&err, C.tensor(t), C.bool(tanh))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func LeakyRelu(t Tensor, negativeSlope float64) Tensor {
	var err *C.char
	ptr := C.tensor_leaky_relu(&err, C.tensor(t), C.double(negativeSlope))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Silu(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_silu(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Sigmoid(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_sigmoid(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Tanh(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_tanh(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Softmax(t Tensor, dim int64) Tensor {
	var err *C.char
	ptr := C.tensor_softmax(&err, C.tensor(t), C.int64_t(dim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Softmax1(t Tensor, dim int64) Tensor {
	var err *C.char
	ptr := C.tensor_softmax1(&err, C.tensor(t), C.int64_t(dim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Dropout(t Tensor, p float64, train bool) Tensor {
	var err *C.char
	ptr := C.tensor_dropout(&err, C.tensor(t), C.double(p), C.bool(train))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Unsqueeze(t Tensor, dim int64) Tensor {
	var err *C.char
	ptr := C.tensor_unsqueeze(&err, C.tensor(t), C.int64_t(dim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Squeeze(t Tensor, dim int64) Tensor {
	var err *C.char
	ptr := C.tensor_squeeze(&err, C.tensor(t), C.int64_t(dim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Flatten(t Tensor, startDim, endDim int64) Tensor {
	var err *C.char
	ptr := C.tensor_flatten(&err, C.tensor(t), C.int64_t(startDim), C.int64_t(endDim))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Contiguous(t Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_contiguous(&err, C.tensor(t))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Expand(t Tensor, sizes []int64) Tensor {
	var err *C.char
	sz, size := cInts[int64, C.int64_t](sizes)
	ptr := C.tensor_expand(&err, C.tensor(t), sz, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Gather(t Tensor, dim int64, index Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_gather(&err, C.tensor(t), C.int64_t(dim), C.tensor(index))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Clamp(t Tensor, min, max float64) Tensor {
	var err *C.char
	ptr := C.tensor_clamp(&err, C.tensor(t), C.double(min), C.double(max))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func MinTensor(a, b Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_min_tensor(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func MaxTensor(a, b Tensor) Tensor {
	var err *C.char
	ptr := C.tensor_max_tensor(&err, C.tensor(a), C.tensor(b))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}
