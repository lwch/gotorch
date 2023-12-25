package tensor

import "github.com/lwch/gotorch/internal/torch"

type conv1DOpt struct {
	stride   int
	padding  int
	dilation int
	groups   int
}

type Conv1DOptFunc func(*conv1DOpt)

func Conv1DStride(v int) Conv1DOptFunc {
	return func(opt *conv1DOpt) {
		opt.stride = v
	}
}

func Conv1DPadding(v int) Conv1DOptFunc {
	return func(opt *conv1DOpt) {
		opt.padding = v
	}
}

func Conv1DDilation(v int) Conv1DOptFunc {
	return func(opt *conv1DOpt) {
		opt.dilation = v
	}
}

func Conv1DGroups(v int) Conv1DOptFunc {
	return func(opt *conv1DOpt) {
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
func (t *Tensor) Conv1D(weight, bias *Tensor, opts ...Conv1DOptFunc) *Tensor {
	opt := &conv1DOpt{
		stride:   1,
		padding:  0,
		dilation: 1,
		groups:   1,
	}
	for _, o := range opts {
		o(opt)
	}
	var b torch.Tensor
	if bias != nil {
		b = bias.t
	}
	ptr := torch.Conv1D(t.t, weight.t, b, opt.stride, opt.padding, opt.dilation, opt.groups)
	return New(ptr)
}

type conv2DOpt struct {
	stride1, stride2   int
	padding1, padding2 int
	dilation           int
	groups             int
}

type Conv2DOptFunc func(*conv2DOpt)

func Conv2DStride(v1, v2 int) Conv2DOptFunc {
	return func(opt *conv2DOpt) {
		opt.stride1 = v1
		opt.stride2 = v2
	}
}

func Conv2DPadding(v1, v2 int) Conv2DOptFunc {
	return func(opt *conv2DOpt) {
		opt.padding1 = v1
		opt.padding2 = v2
	}
}

func Conv2DDilation(v int) Conv2DOptFunc {
	return func(opt *conv2DOpt) {
		opt.dilation = v
	}
}

func Conv2DGroups(v int) Conv2DOptFunc {
	return func(opt *conv2DOpt) {
		opt.groups = v
	}
}

// Conv2D 2D convolution
// weight: [out_channels, in_channels/groups, kernel_size, kernel_size]
// bias: [out_channels]
// stride: stride, default (1, 1)
// padding: padding, default (0, 0)
// dilation: dilation, default 1
// groups: groups, default 1
func (t *Tensor) Conv2D(weight, bias *Tensor, opts ...Conv2DOptFunc) *Tensor {
	opt := &conv2DOpt{
		stride1:  1,
		stride2:  1,
		padding1: 0,
		padding2: 0,
		dilation: 1,
		groups:   1,
	}
	for _, o := range opts {
		o(opt)
	}
	var b torch.Tensor
	if bias != nil {
		b = bias.t
	}
	ptr := torch.Conv2D(t.t, weight.t, b,
		[2]int{opt.stride1, opt.stride2},
		[2]int{opt.padding1, opt.padding2},
		opt.dilation, opt.groups)
	return New(ptr)
}

type conv3DOpt struct {
	stride1, stride2, stride3    int
	padding1, padding2, padding3 int
	dilation                     int
	groups                       int
}

type Conv3DOptFunc func(*conv3DOpt)

func Conv3DStride(v1, v2, v3 int) Conv3DOptFunc {
	return func(opt *conv3DOpt) {
		opt.stride1 = v1
		opt.stride2 = v2
		opt.stride3 = v3
	}
}

func Conv3DPadding(v1, v2, v3 int) Conv3DOptFunc {
	return func(opt *conv3DOpt) {
		opt.padding1 = v1
		opt.padding2 = v2
		opt.padding3 = v3
	}
}

func Conv3DDilation(v int) Conv3DOptFunc {
	return func(opt *conv3DOpt) {
		opt.dilation = v
	}
}

func Conv3DGroups(v int) Conv3DOptFunc {
	return func(opt *conv3DOpt) {
		opt.groups = v
	}
}

// Conv3D 3D convolution
// weight: [out_channels, in_channels/groups, kernel_size, kernel_size, kernel_size]
// bias: [out_channels]
// stride: stride, default (1, 1, 1)
// padding: padding, default (0, 0, 0)
// dilation: dilation, default 1
// groups: groups, default 1
func (t *Tensor) Conv3D(weight, bias *Tensor, opts ...Conv3DOptFunc) *Tensor {
	opt := &conv3DOpt{
		stride1:  1,
		stride2:  1,
		stride3:  1,
		padding1: 0,
		padding2: 0,
		padding3: 0,
		dilation: 1,
		groups:   1,
	}
	for _, o := range opts {
		o(opt)
	}
	var b torch.Tensor
	if bias != nil {
		b = bias.t
	}
	ptr := torch.Conv3D(t.t, weight.t, b,
		[3]int{opt.stride1, opt.stride2, opt.stride3},
		[3]int{opt.padding1, opt.padding2, opt.padding3},
		opt.dilation, opt.groups)
	return New(ptr)
}
