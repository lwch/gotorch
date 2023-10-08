package tensor

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
)

func (t *Tensor) Backward() {
	t.t.Backward(false)
}

func (t *Tensor) BackwardRetained() {
	t.t.Backward(true)
}

func (t *Tensor) store1(ret *torch.Tensor) *mmgr.Storage {
	var s *mmgr.Storage
	if t.s != nil {
		s = t.s
	}
	if s != nil {
		s.Put(ret)
	}
	return s
}

func (t *Tensor) store2(t2 *Tensor, ret *torch.Tensor) *mmgr.Storage {
	var s *mmgr.Storage
	if t.s != nil {
		s = t.s
	} else if t2 != nil && t2.s != nil {
		s = t2.s
	}
	if s != nil {
		s.Put(ret)
	}
	return s
}

func (t *Tensor) store3(t2, t3 *Tensor, ret *torch.Tensor) *mmgr.Storage {
	var s *mmgr.Storage
	if t.s != nil {
		s = t.s
	} else if t2 != nil && t2.s != nil {
		s = t2.s
	} else if t3 != nil && t3.s != nil {
		s = t3.s
	}
	if s != nil {
		s.Put(ret)
	}
	return s
}

func (t *Tensor) MatMul(t2 *Tensor) *Tensor {
	ret := t.t.MatMul(t2.t)
	return &Tensor{s: t.store2(t2, ret), t: ret}
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	ret := t.t.Add(t2.t)
	return &Tensor{s: t.store2(t2, ret), t: ret}
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	ret := t.t.Sub(t2.t)
	return &Tensor{s: t.store2(t2, ret), t: ret}
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	ret := t.t.Mul(t2.t)
	return &Tensor{s: t.store2(t2, ret), t: ret}
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	ret := t.t.Div(t2.t)
	return &Tensor{s: t.store2(t2, ret), t: ret}
}

func (t *Tensor) Pow(n float64) *Tensor {
	ret := t.t.Pow(n)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Sqrt() *Tensor {
	ret := t.t.Sqrt()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) RSqrt() *Tensor {
	ret := t.t.RSqrt()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Log() *Tensor {
	ret := t.t.Log()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Exp() *Tensor {
	ret := t.t.Exp()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Neg() *Tensor {
	ret := t.t.Neg()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Abs() *Tensor {
	ret := t.t.Abs()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Max(dim int64, keepdim bool) *Tensor {
	ret := t.t.Max(dim, keepdim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Min(dim int64, keepdim bool) *Tensor {
	ret := t.t.Min(dim, keepdim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Sum(dim int64, keepdim bool) *Tensor {
	ret := t.t.Sum(dim, keepdim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Mean(dim int64, keepdim bool) *Tensor {
	ret := t.t.Mean(dim, keepdim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Var(dim int64, unbiased, keepdim bool) *Tensor {
	ret := t.t.Var(dim, unbiased, keepdim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Relu() *Tensor {
	ret := t.t.Relu()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Gelu(tanh bool) *Tensor {
	ret := t.t.Gelu(tanh)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) LeakyRelu(negSlope float64) *Tensor {
	ret := t.t.LeakyRelu(negSlope)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Sigmoid() *Tensor {
	ret := t.t.Sigmoid()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Tanh() *Tensor {
	ret := t.t.Tanh()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Softmax(dim int64) *Tensor {
	ret := t.t.Softmax(dim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Softmax1(dim int64) *Tensor {
	ret := t.t.Softmax1(dim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Dropout(p float64, train bool) *Tensor {
	ret := t.t.Dropout(p, train)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Unsqueeze(dim int64) *Tensor {
	ret := t.t.Unsqueeze(dim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Squeeze(dim int64) *Tensor {
	ret := t.t.Squeeze(dim)
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Contiguous() *Tensor {
	ret := t.t.Contiguous()
	return &Tensor{s: t.store1(ret), t: ret}
}

func (t *Tensor) Expand(sizes ...int64) *Tensor {
	ret := t.t.Expand(sizes)
	return &Tensor{s: t.store1(ret), t: ret}
}
