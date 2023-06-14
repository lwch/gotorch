package tensor

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
)

func ARange(s *mmgr.Storage, n int, dtype consts.ScalarType) *Tensor {
	t := torch.ARange(n, dtype)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func Zeros(s *mmgr.Storage, dtype consts.ScalarType, shape ...int64) *Tensor {
	t := torch.Zeros(shape, dtype)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromUint8(s *mmgr.Storage, data []uint8, shape ...int64) *Tensor {
	t := torch.FromUint8(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromInt8(s *mmgr.Storage, data []int8, shape ...int64) *Tensor {
	t := torch.FromInt8(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromInt16(s *mmgr.Storage, data []int16, shape ...int64) *Tensor {
	t := torch.FromInt16(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromInt32(s *mmgr.Storage, data []int32, shape ...int64) *Tensor {
	t := torch.FromInt32(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromInt64(s *mmgr.Storage, data []int64, shape ...int64) *Tensor {
	t := torch.FromInt64(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromFloat32(s *mmgr.Storage, data []float32, shape ...int64) *Tensor {
	t := torch.FromFloat32(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromFloat64(s *mmgr.Storage, data []float64, shape ...int64) *Tensor {
	t := torch.FromFloat64(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func FromBool(s *mmgr.Storage, data []bool, shape ...int64) *Tensor {
	t := torch.FromBool(data, shape)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func VStack(a, b *Tensor) *Tensor {
	var s *mmgr.Storage
	if a.s != nil {
		s = a.s
	} else if b.s != nil {
		s = b.s
	}
	t := torch.VStack(a.t, b.t)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func HStack(a, b *Tensor) *Tensor {
	var s *mmgr.Storage
	if a.s != nil {
		s = a.s
	} else if b.s != nil {
		s = b.s
	}
	t := torch.HStack(a.t, b.t)
	if s != nil {
		s.Put(t)
	}
	return &Tensor{s: s, t: t}
}

func (t *Tensor) Uint8Value() []uint8 {
	return t.t.Uint8Value()
}

func (t *Tensor) Int8Value() []int8 {
	return t.t.Int8Value()
}

func (t *Tensor) Int16Value() []int16 {
	return t.t.Int16Value()
}

func (t *Tensor) Int32Value() []int32 {
	return t.t.Int32Value()
}

func (t *Tensor) Int64Value() []int64 {
	return t.t.Int64Value()
}

func (t *Tensor) Float32Value() []float32 {
	return t.t.Float32Value()
}

func (t *Tensor) Float64Value() []float64 {
	return t.t.Float64Value()
}

func (t *Tensor) BoolValue() []bool {
	return t.t.BoolValue()
}

func (t *Tensor) NArrow(dim, start, length int64) *Tensor {
	ptr := t.t.NArrow(dim, start, length)
	if t.s != nil {
		t.s.Put(ptr)
	}
	return &Tensor{s: t.s, t: ptr}
}

func (t *Tensor) View(shapes ...int64) *Tensor {
	ptr := t.t.View(shapes)
	if t.s != nil {
		t.s.Put(ptr)
	}
	return &Tensor{s: t.s, t: ptr}
}

func (t *Tensor) Permute(dims ...int64) *Tensor {
	ptr := t.t.Permute(dims)
	if t.s != nil {
		t.s.Put(ptr)
	}
	return &Tensor{s: t.s, t: ptr}
}
