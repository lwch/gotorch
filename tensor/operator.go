package tensor

import (
	"fmt"

	"github.com/lwch/gotorch/internal/torch"
)

func (t *Tensor) Backward() {
	torch.Backward(t.t, false)
}

func (t *Tensor) BackwardRetained() {
	torch.Backward(t.t, true)
}

func (t *Tensor) MatMul(t2 *Tensor) *Tensor {
	ptr := torch.MatMul(t.t, t2.t)
	return New(ptr, fmt.Sprintf("%v@%v", t.name, t2.name))
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	ptr := torch.Add(t.t, t2.t)
	return New(ptr, fmt.Sprintf("%v+%v", t.name, t2.name))
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	ptr := torch.Sub(t.t, t2.t)
	return New(ptr, fmt.Sprintf("%v-%v", t.name, t2.name))
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	ptr := torch.Mul(t.t, t2.t)
	return New(ptr, fmt.Sprintf("%v*%v", t.name, t2.name))
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	ptr := torch.Div(t.t, t2.t)
	return New(ptr, fmt.Sprintf("%v/%v", t.name, t2.name))
}

func (t *Tensor) Pow(n float64) *Tensor {
	ptr := torch.Pow(t.t, n)
	return New(ptr, fmt.Sprintf("%v^%v", t.name, n))
}

func (t *Tensor) Sqrt() *Tensor {
	ptr := torch.Sqrt(t.t)
	return New(ptr, fmt.Sprintf("sqrt(%v)", t.name))
}

func (t *Tensor) RSqrt() *Tensor {
	ptr := torch.RSqrt(t.t)
	return New(ptr, fmt.Sprintf("rsqrt(%v)", t.name))
}

func (t *Tensor) Log() *Tensor {
	ptr := torch.Log(t.t)
	return New(ptr, fmt.Sprintf("log(%v)", t.name))
}

func (t *Tensor) Exp() *Tensor {
	ptr := torch.Exp(t.t)
	return New(ptr, fmt.Sprintf("exp(%v)", t.name))
}

func (t *Tensor) Neg() *Tensor {
	ptr := torch.Neg(t.t)
	return New(ptr, fmt.Sprintf("-%v", t.name))
}

func (t *Tensor) Abs() *Tensor {
	ptr := torch.Abs(t.t)
	return New(ptr, fmt.Sprintf("abs(%v)", t.name))
}

func (t *Tensor) Max(dim int64, keepdim bool) *Tensor {
	ptr := torch.Max(t.t, dim, keepdim)
	return New(ptr, fmt.Sprintf("max(%v)", t.name))
}

func (t *Tensor) Min(dim int64, keepdim bool) *Tensor {
	ptr := torch.Min(t.t, dim, keepdim)
	return New(ptr, fmt.Sprintf("min(%v)", t.name))
}

func (t *Tensor) Sum(dim int64, keepdim bool) *Tensor {
	ptr := torch.Sum(t.t, dim, keepdim)
	return New(ptr, fmt.Sprintf("sum(%v)", t.name))
}

func (t *Tensor) Mean(dim int64, keepdim bool) *Tensor {
	ptr := torch.Mean(t.t, dim, keepdim)
	return New(ptr, fmt.Sprintf("mean(%v)", t.name))
}

func (t *Tensor) Var(dim int64, unbiased, keepdim bool) *Tensor {
	ptr := torch.Var(t.t, dim, unbiased, keepdim)
	return New(ptr, fmt.Sprintf("var(%v)", t.name))
}

func (t *Tensor) Relu() *Tensor {
	ptr := torch.Relu(t.t)
	return New(ptr, fmt.Sprintf("relu(%v)", t.name))
}

func (t *Tensor) Gelu(tanh bool) *Tensor {
	ptr := torch.Gelu(t.t, tanh)
	return New(ptr, fmt.Sprintf("gelu(%v)", t.name))
}

func (t *Tensor) LeakyRelu(negSlope float64) *Tensor {
	ptr := torch.LeakyRelu(t.t, negSlope)
	return New(ptr, fmt.Sprintf("leaky_relu(%v)", t.name))
}

func (t *Tensor) Silu() *Tensor {
	ptr := torch.Silu(t.t)
	return New(ptr, fmt.Sprintf("silu(%v)", t.name))
}

func (t *Tensor) Sigmoid() *Tensor {
	ptr := torch.Sigmoid(t.t)
	return New(ptr, fmt.Sprintf("sigmoid(%v)", t.name))
}

func (t *Tensor) Tanh() *Tensor {
	ptr := torch.Tanh(t.t)
	return New(ptr, fmt.Sprintf("tanh(%v)", t.name))
}

func (t *Tensor) Softmax(dim int64) *Tensor {
	ptr := torch.Softmax(t.t, dim)
	return New(ptr, fmt.Sprintf("softmax(%v)", t.name))
}

func (t *Tensor) Softmax1(dim int64) *Tensor {
	ptr := torch.Softmax1(t.t, dim)
	return New(ptr, fmt.Sprintf("softmax1(%v)", t.name))
}

func (t *Tensor) Dropout(p float64, train bool) *Tensor {
	ptr := torch.Dropout(t.t, p, train)
	return New(ptr, fmt.Sprintf("dropout(%v)", t.name))
}

func (t *Tensor) Unsqueeze(dim int64) *Tensor {
	ptr := torch.Unsqueeze(t.t, dim)
	return New(ptr, fmt.Sprintf("unsqueeze(%v)", t.name))
}

func (t *Tensor) Squeeze(dim int64) *Tensor {
	ptr := torch.Squeeze(t.t, dim)
	return New(ptr, fmt.Sprintf("squeeze(%v)", t.name))
}

func (t *Tensor) Contiguous() *Tensor {
	ptr := torch.Contiguous(t.t)
	return New(ptr, fmt.Sprintf("contiguous(%v)", t.name))
}

func (t *Tensor) Expand(sizes ...int64) *Tensor {
	ptr := torch.Expand(t.t, sizes)
	return New(ptr, fmt.Sprintf("expand(%v)", t.name))
}
