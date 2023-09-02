package tensor

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
)

type Tensor struct {
	s *mmgr.Storage
	t *torch.Tensor
}

func New(t *torch.Tensor, s *mmgr.Storage) *Tensor {
	return &Tensor{
		t: t,
		s: s,
	}
}

func (t *Tensor) Storage() *mmgr.Storage {
	return t.s
}

func (t *Tensor) Tensor() *torch.Tensor {
	return t.t
}

func (t *Tensor) UnFree() {
	if t.s != nil {
		t.s.Remove(t.t)
	}
}

func (t *Tensor) Free() {
	t.t.Free()
}

func (t *Tensor) Reshape(shape ...int64) *Tensor {
	ret := t.t.Reshape(shape)
	if t.s != nil {
		t.s.Put(ret)
	}
	return &Tensor{s: t.s, t: ret}
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ret := t.t.Transpose(dim1, dim2)
	if t.s != nil {
		t.s.Put(ret)
	}
	return &Tensor{s: t.s, t: ret}
}

func (t *Tensor) ElemSize() int64 {
	return t.t.ElemSize()
}

func (t *Tensor) ElemCount() int64 {
	return t.t.ElemCount()
}

func (t *Tensor) Dims() int64 {
	return t.t.Dims()
}

func (t *Tensor) Shapes() []int64 {
	return t.t.Shapes()
}

func (t *Tensor) ScalarType() consts.ScalarType {
	return t.t.ScalarType()
}

func (t *Tensor) SetRequiresGrad(b bool) {
	t.t.SetRequiresGrad(b)
}

func (t *Tensor) ToDevice(device consts.DeviceType) *Tensor {
	ret := t.t.ToDevice(device)
	if t.s != nil {
		t.s.Put(ret)
	}
	return &Tensor{s: t.s, t: ret}
}
