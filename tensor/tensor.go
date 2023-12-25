package tensor

import (
	"fmt"
	"runtime"
	"time"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/logging"
)

type Tensor struct {
	name    string
	created time.Time
	t       torch.Tensor
}

func New(t torch.Tensor, name string) *Tensor {
	ts := &Tensor{
		name:    name,
		created: time.Now(),
		t:       t,
	}
	logging.Debug("new tensor: %p", ts)
	runtime.SetFinalizer(ts, freeTensor)
	logBuildInfo(ts)
	return ts
}

func freeTensor(t *Tensor) error {
	if t == nil || t.t == nil {
		return nil
	}
	logging.Debug("free tensor: %p", t)
	torch.FreeTensor(t.t)
	t.t = nil
	free(t)
	runtime.SetFinalizer(t, nil)
	return nil
}

func (t *Tensor) Created() time.Time {
	return t.created
}

func (t *Tensor) Tensor() torch.Tensor {
	return t.t
}

func (t *Tensor) Name() string {
	return t.name
}

func (t *Tensor) Reshape(shape ...int64) *Tensor {
	ptr := torch.Reshape(t.t, shape)
	return New(ptr, fmt.Sprintf("%s.reshape%v", t.name, shape))
}

func (t *Tensor) Transpose(dim1, dim2 int64) *Tensor {
	ptr := torch.Transpose(t.t, dim1, dim2)
	return New(ptr, fmt.Sprintf("%s.transpose[%d,%d]", t.name, dim1, dim2))
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
	return New(ptr, fmt.Sprintf("%s.to(%s)", t.name, device))
}

func (t *Tensor) ToScalarType(scalarType consts.ScalarType) *Tensor {
	ptr := torch.ToScalarType(t.t, scalarType)
	return New(ptr, fmt.Sprintf("%s.to(%s)", t.name, scalarType))
}
