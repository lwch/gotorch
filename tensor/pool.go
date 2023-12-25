package tensor

import (
	"fmt"

	"github.com/lwch/gotorch/internal/torch"
)

type poolOpt struct {
	stride   int
	padding  int
	dilation int
	ceil     bool
}

// PoolOpt is an option for pooling operations.
type PoolOpt func(*poolOpt)

// PoolStride sets the stride for pooling operations.
func PoolStride(s int) PoolOpt {
	return func(o *poolOpt) {
		o.stride = s
	}
}

// PoolPadding sets the padding for pooling operations.
func PoolPadding(p int) PoolOpt {
	return func(o *poolOpt) {
		o.padding = p
	}
}

// PoolDilation sets the dilation for pooling operations.
func PoolDilation(d int) PoolOpt {
	return func(o *poolOpt) {
		o.dilation = d
	}
}

// PoolCeil sets the ceil for pooling operations.
func PoolCeil(c bool) PoolOpt {
	return func(o *poolOpt) {
		o.ceil = c
	}
}

// MaxPool1D returns a new tensor with the result of applying a 1D max pooling
// operation on the input tensor.
// kernel: kernel size
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// ceil: ceil, default false
func (t *Tensor) MaxPool1D(kernel int, opt ...PoolOpt) *Tensor {
	p := &poolOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		ceil:     false,
	}
	for _, o := range opt {
		o(p)
	}
	ptr := torch.MaxPool1D(t.t, kernel, p.stride, p.padding, p.dilation, p.ceil)
	return New(ptr, fmt.Sprintf("MaxPool1D(%s)", t.name))
}

// MaxPool2D returns a new tensor with the result of applying a 2D max pooling
// operation on the input tensor.
// kernel: kernel size
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// ceil: ceil, default false
func (t *Tensor) MaxPool2D(kernel int, opt ...PoolOpt) *Tensor {
	p := &poolOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		ceil:     false,
	}
	for _, o := range opt {
		o(p)
	}
	ptr := torch.MaxPool2D(t.t, kernel, p.stride, p.padding, p.dilation, p.ceil)
	return New(ptr, fmt.Sprintf("MaxPool2D(%s)", t.name))
}

// MaxPool3D returns a new tensor with the result of applying a 3D max pooling
// operation on the input tensor.
// kernel: kernel size
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// ceil: ceil, default false
func (t *Tensor) MaxPool3D(kernel int, opt ...PoolOpt) *Tensor {
	p := &poolOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		ceil:     false,
	}
	for _, o := range opt {
		o(p)
	}
	ptr := torch.MaxPool3D(t.t, kernel, p.stride, p.padding, p.dilation, p.ceil)
	return New(ptr, fmt.Sprintf("MaxPool3D(%s)", t.name))
}

// AvgPool1D returns a new tensor with the result of applying a 1D average pooling
// operation on the input tensor.
// kernel: kernel size
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// ceil: ceil, default false
func (t *Tensor) AvgPool1D(kernel int, opt ...PoolOpt) *Tensor {
	p := &poolOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		ceil:     false,
	}
	for _, o := range opt {
		o(p)
	}
	ptr := torch.AvgPool1D(t.t, kernel, p.stride, p.padding, p.dilation, p.ceil)
	return New(ptr, fmt.Sprintf("AvgPool1D(%s)", t.name))
}

// AvgPool2D returns a new tensor with the result of applying a 2D average pooling
// operation on the input tensor.
// kernel: kernel size
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// ceil: ceil, default false
func (t *Tensor) AvgPool2D(kernel int, opt ...PoolOpt) *Tensor {
	p := &poolOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		ceil:     false,
	}
	for _, o := range opt {
		o(p)
	}
	ptr := torch.AvgPool2D(t.t, kernel, p.stride, p.padding, p.dilation, p.ceil)
	return New(ptr, fmt.Sprintf("AvgPool2D(%s)", t.name))
}

// AvgPool3D returns a new tensor with the result of applying a 3D average pooling
// operation on the input tensor.
// kernel: kernel size
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// ceil: ceil, default false
func (t *Tensor) AvgPool3D(kernel int, opt ...PoolOpt) *Tensor {
	p := &poolOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		ceil:     false,
	}
	for _, o := range opt {
		o(p)
	}
	ptr := torch.AvgPool3D(t.t, kernel, p.stride, p.padding, p.dilation, p.ceil)
	return New(ptr, fmt.Sprintf("AvgPool3D(%s)", t.name))
}
