package tensor

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
)

func FromUint8(s *mmgr.Storage, data []uint8, shape ...int64) *Tensor {
	t := torch.FromUint8(data, shape)
	s.Put(t)
	return &Tensor{s: s, t: t}
}

func FromInt8(s *mmgr.Storage, data []int8, shape ...int64) *Tensor {
	t := torch.FromInt8(data, shape)
	s.Put(t)
	return &Tensor{s: s, t: t}
}

func FromInt16(s *mmgr.Storage, data []int16, shape ...int64) *Tensor {
	t := torch.FromInt16(data, shape)
	s.Put(t)
	return &Tensor{s: s, t: t}
}

func FromInt32(s *mmgr.Storage, data []int32, shape ...int64) *Tensor {
	t := torch.FromInt32(data, shape)
	s.Put(t)
	return &Tensor{s: s, t: t}
}

func FromInt64(s *mmgr.Storage, data []int64, shape ...int64) *Tensor {
	t := torch.FromInt64(data, shape)
	s.Put(t)
	return &Tensor{s: s, t: t}
}

func FromFloat32(s *mmgr.Storage, data []float32, shape ...int64) *Tensor {
	t := torch.FromFloat32(data, shape)
	s.Put(t)
	return &Tensor{s: s, t: t}
}

func FromFloat64(s *mmgr.Storage, data []float64, shape ...int64) *Tensor {
	t := torch.FromFloat64(data, shape)
	s.Put(t)
	return &Tensor{s: s, t: t}
}

func FromBool(s *mmgr.Storage, data []bool, shape ...int64) *Tensor {
	t := torch.FromBool(data, shape)
	s.Put(t)
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
