package tensor

import (
	"fmt"
	"strings"

	"github.com/lwch/gotorch/internal/torch"
)

func ScaledDotProductAttention(q, k, v, mask *Tensor, drouput float64, isCausal bool) *Tensor {
	var mt torch.Tensor
	var name string
	if mask != nil {
		mt = mask.t
		name = fmt.Sprintf("ScaledDotProductAttention(%v,%v,%v,%v)", q.name, k.name, v.name, mask.name)
	} else {
		name = fmt.Sprintf("ScaledDotProductAttention(%v,%v,%v)", q.name, k.name, v.name)
	}
	ptr := torch.ScaledDotProductAttention(q.t, k.t, v.t, mt, drouput, isCausal)
	return New(ptr, name)
}

func Embedding(input *Tensor, weight *Tensor, paddingIdx int64) *Tensor {
	ptr := torch.TEmbedding(input.t, weight.t, paddingIdx)
	return New(ptr, fmt.Sprintf("Embedding(%v,%v)", input.name, weight.name))
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
	names := make([]string, len(tensors))
	for i, t := range tensors {
		list[i] = t.t
		names[i] = t.name
	}
	ptr := torch.Cat(list, dim)
	return New(ptr, fmt.Sprintf("Cat(%s)", strings.Join(names, ",")))
}

func SVD(t *Tensor) (*Tensor, *Tensor, *Tensor) {
	u, s, v := torch.SVD(t.t)
	return New(u, fmt.Sprintf("U(%v)", t.name)),
		New(s, fmt.Sprintf("S(%v)", t.name)),
		New(v, fmt.Sprintf("V(%v)", t.name))
}

func Outer(a, b *Tensor) *Tensor {
	ptr := torch.Outer(a.t, b.t)
	return New(ptr, fmt.Sprintf("Outer(%v,%v)", a.name, b.name))
}

func Polar(abs, angle *Tensor) *Tensor {
	ptr := torch.Polar(abs.t, angle.t)
	return New(ptr, fmt.Sprintf("Polar(%v,%v)", abs.name, angle.name))
}

func (t *Tensor) ViewAsComplex() *Tensor {
	ptr := torch.ViewAsComplex(t.t)
	return New(ptr, fmt.Sprintf("ViewAsComplex(%v)", t.name))
}

func (t *Tensor) ViewAsReal() *Tensor {
	ptr := torch.ViewAsReal(t.t)
	return New(ptr, fmt.Sprintf("ViewAsReal(%v)", t.name))
}
