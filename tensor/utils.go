package tensor

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
)

func ScaledDotProductAttention(q, k, v, mask *Tensor, drouput float64, isCausal bool) *Tensor {
	var mt *torch.Tensor
	if mask != nil {
		mt = mask.t
	}
	ret := torch.ScaledDotProductAttention(q.t, k.t, v.t, mt, drouput, isCausal)
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
	}
	return &Tensor{s: store, t: ret}
}

func Embedding(input *Tensor, weight *Tensor, paddingIdx int64) *Tensor {
	ret := torch.TEmbedding(input.t, weight.t, paddingIdx)
	var store *mmgr.Storage
	if input.s != nil {
		store = input.s
	} else if weight.s != nil {
		store = weight.s
	}
	if store != nil {
		store.Put(ret)
	}
	return &Tensor{s: store, t: ret}
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

func Cat(tensors []*Tensor, dim int) *Tensor {
	list := make([]*torch.Tensor, len(tensors))
	for i, t := range tensors {
		list[i] = t.t
	}
	ret := torch.Cat(list, dim)
	var store *mmgr.Storage
	for _, t := range tensors {
		if t.s != nil {
			store = t.s
			break
		}
	}
	if store != nil {
		store.Put(ret)
	}
	return &Tensor{s: store, t: ret}
}

func SVD(t *Tensor) (*Tensor, *Tensor, *Tensor) {
	u, s, v := torch.SVD(t.t)
	var store *mmgr.Storage
	if t.s != nil {
		store = t.s
	}
	if store != nil {
		store.Put(u)
		store.Put(s)
		store.Put(v)
	}
	return &Tensor{s: store, t: u}, &Tensor{s: store, t: s}, &Tensor{s: store, t: v}
}
