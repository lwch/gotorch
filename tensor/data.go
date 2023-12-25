package tensor

import (
	"fmt"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
)

func ARange(name string, n int, dtype consts.ScalarType, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.ARange(n, dtype, args.device)
	return New(ptr, name)
}

func Zeros(name string, dtype consts.ScalarType, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.Zeros(args.shapes, dtype, args.device)
	return New(ptr, name)
}

func FromUint8(name string, data []uint8, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromUint8(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromInt8(name string, data []int8, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromInt8(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromInt16(name string, data []int16, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromInt16(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromInt32(name string, data []int32, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromInt32(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromInt64(name string, data []int64, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromInt64(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromHalf(name string, data []float32, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromHalf(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromHalfRaw(name string, data []uint16, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromHalfRaw(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromBFloat16(name string, data []float32, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromBFloat16(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromBFloat16Raw(name string, data []uint16, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromBFloat16Raw(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromFloat32(name string, data []float32, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromFloat32(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromFloat64(name string, data []float64, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromFloat64(data, args.shapes, args.device)
	return New(ptr, name)
}

func FromBool(name string, data []bool, opts ...Option) *Tensor {
	args := defaultOptions()
	for _, opt := range opts {
		opt(args)
	}
	ptr := torch.FromBool(data, args.shapes, args.device)
	return New(ptr, name)
}

func VStack(a, b *Tensor) *Tensor {
	ptr := torch.VStack(a.t, b.t)
	return New(ptr, fmt.Sprintf("%s.vstack(%s)", a.name, b.name))
}

func HStack(a, b *Tensor) *Tensor {
	ptr := torch.HStack(a.t, b.t)
	return New(ptr, fmt.Sprintf("%s.hstack(%s)", a.name, b.name))
}

func (t *Tensor) Uint8Value() []uint8 {
	return torch.Uint8Value(t.t)
}

func (t *Tensor) Int8Value() []int8 {
	return torch.Int8Value(t.t)
}

func (t *Tensor) Int16Value() []int16 {
	return torch.Int16Value(t.t)
}

func (t *Tensor) Int32Value() []int32 {
	return torch.Int32Value(t.t)
}

func (t *Tensor) Int64Value() []int64 {
	return torch.Int64Value(t.t)
}

func (t *Tensor) HalfValue() []float32 {
	return torch.HalfValue(t.t)
}

func (t *Tensor) HalfRaw() []uint16 {
	return torch.HalfRaw(t.t)
}

func (t *Tensor) BFloat16Value() []float32 {
	return torch.BFloat16Value(t.t)
}

func (t *Tensor) BFloat16Raw() []uint16 {
	return torch.BFloat16Raw(t.t)
}

func (t *Tensor) Float32Value() []float32 {
	return torch.Float32Value(t.t)
}

func (t *Tensor) Float64Value() []float64 {
	return torch.Float64Value(t.t)
}

func (t *Tensor) BoolValue() []bool {
	return torch.BoolValue(t.t)
}

func (t *Tensor) NArrow(dim, start, length int64) *Tensor {
	ptr := torch.NArrow(t.t, dim, start, length)
	return New(ptr, fmt.Sprintf("%s.narrow(%d,%d,%d)", t.name, dim, start, length))
}

func (t *Tensor) View(shapes ...int64) *Tensor {
	ptr := torch.View(t.t, shapes)
	return New(ptr, fmt.Sprintf("%s.view%v", t.name, shapes))
}

func (t *Tensor) Permute(dims ...int64) *Tensor {
	ptr := torch.Permute(t.t, dims)
	return New(ptr, fmt.Sprintf("%s.permute%v", t.name, dims))
}
