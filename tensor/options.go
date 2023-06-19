package tensor

import "github.com/lwch/gotorch/consts"

type options struct {
	shapes []int64
	device consts.DeviceType
}

func defaultOptions() *options {
	return &options{
		device: consts.KCPU,
	}
}

type Option func(*options)

func WithShapes(shapes ...int64) Option {
	return func(opts *options) {
		opts.shapes = shapes
	}
}

func WithDevice(device consts.DeviceType) Option {
	return func(opts *options) {
		opts.device = device
	}
}
