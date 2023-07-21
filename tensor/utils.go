package tensor

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
)

func ScaledDotProductAttention(q, k, v, mask *Tensor, drouput float64, isCausal bool) (*Tensor, *Tensor) {
	var mt *torch.Tensor
	if mask != nil {
		mt = mask.t
	}
	ret, score := torch.ScaledDotProductAttention(q.t, k.t, v.t, mt, drouput, isCausal)
	var store *mmgr.Storage
	if q.s != nil {
		store = q.s
	} else if k.s != nil {
		store = k.s
	} else if v.s != nil {
		store = v.s
	} else if mask != nil && mask.s != nil {
		store = mask.s
	}
	if store != nil {
		store.Put(ret)
		store.Put(score)
	}
	return &Tensor{s: store, t: ret}, &Tensor{s: store, t: score}
}

func ClipGradNorm(params []*Tensor, max, t float64) {
	list := make([]*torch.Tensor, len(params))
	for i, p := range params {
		list[i] = p.t
	}
	torch.ClipGradNorm(list, max, t)
}

func (t *Tensor) Print() {
	t.t.Print()
}
