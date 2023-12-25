package nn

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type Attention struct {
	module
}

func NewAttention(embedDim, numHeads int64, dropout float64) *Attention {
	return &Attention{
		module{torch.NewAttention(embedDim, numHeads, dropout)},
	}
}

func (a *Attention) Forward(q, k, v, mask *tensor.Tensor, isCausal bool) (*tensor.Tensor, *tensor.Tensor) {
	var m torch.Tensor
	if mask != nil {
		m = mask.Tensor()
	}
	ret, score := a.m.(torch.AttentionForward).Forward(q.Tensor(), k.Tensor(), v.Tensor(), m, isCausal)
	return tensor.New(ret), tensor.New(score)
}
