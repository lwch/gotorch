package tensor

import (
	"github.com/lwch/gotorch/internal/torch"
)

func ScaledDotProductAttention(q, k, v, mask *Tensor, drouput float64, isCausal bool) *Tensor {
	var mt torch.Tensor
	if mask != nil {
		mt = mask.t
	}
	ptr := torch.ScaledDotProductAttention(q.t, k.t, v.t, mt, drouput, isCausal)
	return New(ptr)
}

func Embedding(input *Tensor, weight *Tensor, paddingIdx int64) *Tensor {
	ptr := torch.TEmbedding(input.t, weight.t, paddingIdx)
	return New(ptr)
}

func ClipGradNorm(params []*Tensor, max, t float64) {
	list := make([]torch.Tensor, len(params))
	for i, p := range params {
		list[i] = p.t
	}
	torch.ClipGradNorm(list, max, t)
}

func (t *Tensor) Print() {
	torch.Print(t.t)
}

func Cat(tensors []*Tensor, dim int) *Tensor {
	list := make([]torch.Tensor, len(tensors))
	for i, t := range tensors {
		list[i] = t.t
	}
	ptr := torch.Cat(list, dim)
	return New(ptr)
}

func SVD(t *Tensor) (*Tensor, *Tensor, *Tensor) {
	u, s, v := torch.SVD(t.t)
	return New(u), New(s), New(v)
}

func Outer(a, b *Tensor) *Tensor {
	ptr := torch.Outer(a.t, b.t)
	return New(ptr)
}

func Polar(abs, angle *Tensor) *Tensor {
	ptr := torch.Polar(abs.t, angle.t)
	return New(ptr)
}

func (t *Tensor) ViewAsComplex() *Tensor {
	ptr := torch.ViewAsComplex(t.t)
	return New(ptr)
}

func (t *Tensor) ViewAsReal() *Tensor {
	ptr := torch.ViewAsReal(t.t)
	return New(ptr)
}
