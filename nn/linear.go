package nn

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/tensor"
)

type Linear struct {
	m torch.Module
}

func NewLinear(inFeatures, outFeatures int64) *Linear {
	return &Linear{
		m: torch.NewLinear(inFeatures, outFeatures),
	}
}

func (l *Linear) Parameters(s *mmgr.Storage) []*tensor.Tensor {
	params := l.m.Parameters()
	ret := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		ret[i] = tensor.New(p, s)
	}
	return ret
}

func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	t := l.m.Forward(x.Tensor())
	return tensor.New(t, x.Storage())
}
