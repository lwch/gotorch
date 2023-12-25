package tensor

import (
	"runtime"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/logging"
)

type Tensor struct {
	t torch.Tensor
}

func New(t torch.Tensor) *Tensor {
	ts := &Tensor{t: t}
	logging.Debug("new tensor: %p", ts)
	runtime.SetFinalizer(ts, freeTensor)
	return ts
}

func freeTensor(t *Tensor) error {
	if t == nil || t.t == nil {
		return nil
	}
	logging.Debug("free tensor: %p", t)
	torch.FreeTensor(t.t)
	t.t = nil
	runtime.SetFinalizer(t, nil)
	return nil
}

func (t *Tensor) Tensor() torch.Tensor {
	return t.t
}

func (t *Tensor) Reshape(shape ...int64) *Tensor {
	ptr := torch.Reshape(t.t, shape)
	return New(ptr)
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ptr := torch.Transpose(t.t, dim1, dim2)
	return New(ptr)
}

func (t *Tensor) ElemSize() int64 {
	return torch.ElemSize(t.t)
}

func (t *Tensor) ElemCount() int64 {
	return torch.ElemCount(t.t)
}

func (t *Tensor) Dims() int64 {
	return torch.Dims(t.t)
}

func (t *Tensor) Shapes() []int64 {
	return torch.Shapes(t.t)
}

func (t *Tensor) ScalarType() consts.ScalarType {
	return torch.ScalarType(t.t)
}

func (t *Tensor) DeviceType() consts.DeviceType {
	return torch.DeviceType(t.t)
}

func (t *Tensor) SetRequiresGrad(b bool) {
	torch.SetRequiresGrad(t.t, b)
}

func (t *Tensor) ToDevice(device consts.DeviceType) *Tensor {
	ptr := torch.ToDevice(t.t, device)
	return New(ptr)
}

func (t *Tensor) ToScalarType(scalarType consts.ScalarType) *Tensor {
	ptr := torch.ToScalarType(t.t, scalarType)
	return New(ptr)
}
