package torch

// #include "operator.h"
import "C"

func (t *Tensor) Conv1D(weight, bias *Tensor,
	stride, padding, dilation, groups int) *Tensor {
	var err *C.char
	var b C.tensor
	if bias != nil {
		b = bias.data
	}
	ptr := C.tensor_conv1d(&err, t.data, weight.data, b,
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.int64_t(groups))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Conv2D(weight, bias *Tensor,
	stride, padding, dilation, groups int) *Tensor {
	var err *C.char
	var b C.tensor
	if bias != nil {
		b = bias.data
	}
	ptr := C.tensor_conv2d(&err, t.data, weight.data, b,
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.int64_t(groups))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) Conv3D(weight, bias *Tensor,
	stride, padding, dilation, groups int) *Tensor {
	var err *C.char
	var b C.tensor
	if bias != nil {
		b = bias.data
	}
	ptr := C.tensor_conv3d(&err, t.data, weight.data, b,
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.int64_t(groups))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) MaxPool1D(kernel, stride, padding, dilation int, ceil bool) *Tensor {
	var err *C.char
	ptr := C.tensor_max_pool1d(&err, t.data, C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) MaxPool2D(kernel, stride, padding, dilation int, ceil bool) *Tensor {
	var err *C.char
	ptr := C.tensor_max_pool2d(&err, t.data, C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) MaxPool3D(kernel, stride, padding, dilation int, ceil bool) *Tensor {
	var err *C.char
	ptr := C.tensor_max_pool3d(&err, t.data, C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) AvgPool1D(kernel, stride, padding, dilation int, ceil bool) *Tensor {
	var err *C.char
	ptr := C.tensor_avg_pool1d(&err, t.data, C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) AvgPool2D(kernel, stride, padding, dilation int, ceil bool) *Tensor {
	var err *C.char
	ptr := C.tensor_avg_pool2d(&err, t.data, C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (t *Tensor) AvgPool3D(kernel, stride, padding, dilation int, ceil bool) *Tensor {
	var err *C.char
	ptr := C.tensor_avg_pool3d(&err, t.data, C.int64_t(kernel),
		C.int64_t(stride), C.int64_t(padding),
		C.int64_t(dilation), C.bool(ceil))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}
