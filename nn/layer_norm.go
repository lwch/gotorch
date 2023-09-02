package nn

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/tensor"
)

type LayerNorm struct {
	m torch.Module
}

func NewLayerNorm(shapes ...int64) *LayerNorm {
	return &LayerNorm{
		m: torch.NewLayerNorm(shapes),
	}
}

func (l *LayerNorm) Parameters(s *mmgr.Storage) []*tensor.Tensor {
	params := l.m.Parameters()
	ret := make([]*tensor.Tensor, len(params))
	for i, p := range params {
		ret[i] = tensor.New(p, s)
	}
	return ret
}

func (l *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	t := l.m.Forward(x.Tensor())
	return tensor.New(t, x.Storage())
}
