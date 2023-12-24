package torch

// #include "operator.h"
import "C"

func Conv1D(t, weight, bias Tensor,
	stride, padding, dilation, groups int) Tensor {
	var err *C.char
	ptr := C.tensor_conv1d(&err, C.tensor(t), C.tensor(weight), C.tensor(bias),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.int64_t(groups))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Conv2D(t, weight, bias Tensor,
	stride, padding [2]int, dilation, groups int) Tensor {
	var err *C.char
	var stride1, stride2 C.int64_t
	stride1 = C.int64_t(stride[0])
	stride2 = C.int64_t(stride[1])
	var padding1, padding2 C.int64_t
	padding1 = C.int64_t(padding[0])
	padding2 = C.int64_t(padding[1])
	ptr := C.tensor_conv2d(&err, C.tensor(t), C.tensor(weight), C.tensor(bias),
		stride1, stride2,
		padding1, padding2,
		C.int64_t(dilation), C.int64_t(groups))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Conv3D(t, weight, bias Tensor,
	stride, padding [3]int, dilation, groups int) Tensor {
	var err *C.char
	var stride1, stride2, stride3 C.int64_t
	stride1 = C.int64_t(stride[0])
	stride2 = C.int64_t(stride[1])
	stride3 = C.int64_t(stride[2])
	var padding1, padding2, padding3 C.int64_t
	padding1 = C.int64_t(padding[0])
	padding2 = C.int64_t(padding[1])
	padding3 = C.int64_t(padding[2])
	ptr := C.tensor_conv3d(&err, C.tensor(t), C.tensor(weight), C.tensor(bias),
		stride1, stride2, stride3,
		padding1, padding2, padding3,
		C.int64_t(dilation), C.int64_t(groups))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func MaxPool1D(t Tensor, kernel, stride, padding, dilation int, ceil bool) Tensor {
	var err *C.char
	ptr := C.tensor_max_pool1d(&err, C.tensor(t), C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func MaxPool2D(t Tensor, kernel, stride, padding, dilation int, ceil bool) Tensor {
	var err *C.char
	ptr := C.tensor_max_pool2d(&err, C.tensor(t), C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func MaxPool3D(t Tensor, kernel, stride, padding, dilation int, ceil bool) Tensor {
	var err *C.char
	ptr := C.tensor_max_pool3d(&err, C.tensor(t), C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func AvgPool1D(t Tensor, kernel, stride, padding, dilation int, ceil bool) Tensor {
	var err *C.char
	ptr := C.tensor_avg_pool1d(&err, C.tensor(t), C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func AvgPool2D(t Tensor, kernel, stride, padding, dilation int, ceil bool) Tensor {
	var err *C.char
	ptr := C.tensor_avg_pool2d(&err, C.tensor(t), C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func AvgPool3D(t Tensor, kernel, stride, padding, dilation int, ceil bool) Tensor {
	var err *C.char
	ptr := C.tensor_avg_pool3d(&err, C.tensor(t), C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}
