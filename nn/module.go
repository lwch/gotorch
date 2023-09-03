package nn

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/tensor"
)

type module struct {
	m torch.Module
}

func (m *module) Parameters(s *mmgr.Storage) []*tensor.Tensor {
	params := m.m.Parameters()
	ret := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		ret[i] = tensor.New(p, s)
	}
	return ret
}

func (m *module) Forward(x *tensor.Tensor) *tensor.Tensor {
	t := m.m.(torch.NormalForward).Forward(x.Tensor())
	return tensor.New(t, x.Storage())
}

func (m *module) ToDevice(device consts.DeviceType) {
	m.m.ToDevice(device)
}

func (m *module) ToScalarType(t consts.ScalarType) {
	m.m.ToScalarType(t)
}
