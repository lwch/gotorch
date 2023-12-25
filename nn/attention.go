package nn

import (
	"fmt"

	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type Attention struct {
	module
}

func NewAttention(name string, embedDim, numHeads int64, dropout float64) *Attention {
	return &Attention{
		module{
			name: name,
			m:    torch.NewAttention(embedDim, numHeads, dropout),
		},
	}
}

func (a *Attention) Forward(q, k, v, mask *tensor.Tensor, isCausal bool) (*tensor.Tensor, *tensor.Tensor) {
	var m torch.Tensor
	if mask != nil {
		m = mask.Tensor()
	}
	ret, score := a.m.(torch.AttentionForward).Forward(q.Tensor(), k.Tensor(), v.Tensor(), m, isCausal)
	return tensor.New(ret, fmt.Sprintf("%s.value", a.name)),
		tensor.New(score, fmt.Sprintf("%s.score", a.name))
}
