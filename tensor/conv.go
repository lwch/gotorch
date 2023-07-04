package tensor

import "github.com/lwch/gotorch/internal/torch"

type convOpt struct {
	stride   int
	padding  int
	dilation int
	groups   int
}

type ConvOptFunc func(*convOpt)

func ConvStride(v int) ConvOptFunc {
	return func(opt *convOpt) {
		opt.stride = v
	}
}

func ConvPadding(v int) ConvOptFunc {
	return func(opt *convOpt) {
		opt.padding = v
	}
}

func ConvDilation(v int) ConvOptFunc {
	return func(opt *convOpt) {
		opt.dilation = v
	}
}

func ConvGroups(v int) ConvOptFunc {
	return func(opt *convOpt) {
		opt.groups = v
	}
}

// Conv1D 1D convolution
// weight: [out_channels, in_channels/groups, kernel_size]
// bias: [out_channels]
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// groups: groups, default 1
func (t *Tensor) Conv1D(weight, bias *Tensor, opts ...ConvOptFunc) *Tensor {
	opt := &convOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		groups:   1,
	}
	for _, o := range opts {
		o(opt)
	}
	var b *torch.Tensor
	if bias != nil {
		b = bias.t
	}
	ret := t.t.Conv1D(weight.t, b, opt.stride, opt.padding, opt.dilation, opt.groups)
	return &Tensor{s: t.store3(weight, bias, ret), t: ret}
}

// Conv2D 2D convolution
// weight: [out_channels, in_channels/groups, kernel_size, kernel_size]
// bias: [out_channels]
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// groups: groups, default 1
func (t *Tensor) Conv2D(weight, bias *Tensor, opts ...ConvOptFunc) *Tensor {
	opt := &convOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		groups:   1,
	}
	for _, o := range opts {
		o(opt)
	}
	var b *torch.Tensor
	if bias != nil {
		b = bias.t
	}
	ret := t.t.Conv2D(weight.t, b, opt.stride, opt.padding, opt.dilation, opt.groups)
	return &Tensor{s: t.store3(weight, bias, ret), t: ret}
}

// Conv3D 3D convolution
// weight: [out_channels, in_channels/groups, kernel_size, kernel_size, kernel_size]
// bias: [out_channels]
// stride: stride, default 1
// padding: padding, default 0
// dilation: dilation, default 1
// groups: groups, default 1
func (t *Tensor) Conv3D(weight, bias *Tensor, opts ...ConvOptFunc) *Tensor {
	opt := &convOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		groups:   1,
	}
	for _, o := range opts {
		o(opt)
	}
	var b *torch.Tensor
	if bias != nil {
		b = bias.t
	}
	ret := t.t.Conv3D(weight.t, b, opt.stride, opt.padding, opt.dilation, opt.groups)
	return &Tensor{s: t.store3(weight, bias, ret), t: ret}
}
