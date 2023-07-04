package tensor

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
	ret := t.t.MaxPool1D(kernel, p.stride, p.padding, p.dilation, p.ceil)
	return &Tensor{s: t.store1(ret), t: ret}
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
	ret := t.t.MaxPool2D(kernel, p.stride, p.padding, p.dilation, p.ceil)
	return &Tensor{s: t.store1(ret), t: ret}
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
	ret := t.t.MaxPool3D(kernel, p.stride, p.padding, p.dilation, p.ceil)
	return &Tensor{s: t.store1(ret), t: ret}
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
	ret := t.t.AvgPool1D(kernel, p.stride, p.padding, p.dilation, p.ceil)
	return &Tensor{s: t.store1(ret), t: ret}
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
	ret := t.t.AvgPool2D(kernel, p.stride, p.padding, p.dilation, p.ceil)
	return &Tensor{s: t.store1(ret), t: ret}
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
	ret := t.t.AvgPool3D(kernel, p.stride, p.padding, p.dilation, p.ceil)
	return &Tensor{s: t.store1(ret), t: ret}
}
