package tensor

import (
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
	return New(ptr)
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	ptr := torch.Add(t.t, t2.t)
	return New(ptr)
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	ptr := torch.Sub(t.t, t2.t)
	return New(ptr)
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	ptr := torch.Mul(t.t, t2.t)
	return New(ptr)
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	ptr := torch.Div(t.t, t2.t)
	return New(ptr)
}

func (t *Tensor) Pow(n float64) *Tensor {
	ptr := torch.Pow(t.t, n)
	return New(ptr)
}

func (t *Tensor) Sqrt() *Tensor {
	ptr := torch.Sqrt(t.t)
	return New(ptr)
}

func (t *Tensor) RSqrt() *Tensor {
	ptr := torch.RSqrt(t.t)
	return New(ptr)
}

func (t *Tensor) Log() *Tensor {
	ptr := torch.Log(t.t)
	return New(ptr)
}

func (t *Tensor) Exp() *Tensor {
	ptr := torch.Exp(t.t)
	return New(ptr)
}

func (t *Tensor) Neg() *Tensor {
	ptr := torch.Neg(t.t)
	return New(ptr)
}

func (t *Tensor) Abs() *Tensor {
	ptr := torch.Abs(t.t)
	return New(ptr)
}

func (t *Tensor) Max(dim int64, keepdim bool) *Tensor {
	ptr := torch.Max(t.t, dim, keepdim)
	return New(ptr)
}

func (t *Tensor) Min(dim int64, keepdim bool) *Tensor {
	ptr := torch.Min(t.t, dim, keepdim)
	return New(ptr)
}

func (t *Tensor) Sum(dim int64, keepdim bool) *Tensor {
	ptr := torch.Sum(t.t, dim, keepdim)
	return New(ptr)
}

func (t *Tensor) Mean(dim int64, keepdim bool) *Tensor {
	ptr := torch.Mean(t.t, dim, keepdim)
	return New(ptr)
}

func (t *Tensor) Var(dim int64, unbiased, keepdim bool) *Tensor {
	ptr := torch.Var(t.t, dim, unbiased, keepdim)
	return New(ptr)
}

func (t *Tensor) Relu() *Tensor {
	ptr := torch.Relu(t.t)
	return New(ptr)
}

func (t *Tensor) Gelu(tanh bool) *Tensor {
	ptr := torch.Gelu(t.t, tanh)
	return New(ptr)
}

func (t *Tensor) LeakyRelu(negSlope float64) *Tensor {
	ptr := torch.LeakyRelu(t.t, negSlope)
	return New(ptr)
}

func (t *Tensor) Silu() *Tensor {
	ptr := torch.Silu(t.t)
	return New(ptr)
}

func (t *Tensor) Sigmoid() *Tensor {
	ptr := torch.Sigmoid(t.t)
	return New(ptr)
}

func (t *Tensor) Tanh() *Tensor {
	ptr := torch.Tanh(t.t)
	return New(ptr)
}

func (t *Tensor) Softmax(dim int64) *Tensor {
	ptr := torch.Softmax(t.t, dim)
	return New(ptr)
}

func (t *Tensor) Softmax1(dim int64) *Tensor {
	ptr := torch.Softmax1(t.t, dim)
	return New(ptr)
}

func (t *Tensor) Dropout(p float64, train bool) *Tensor {
	ptr := torch.Dropout(t.t, p, train)
	return New(ptr)
}

func (t *Tensor) Unsqueeze(dim int64) *Tensor {
	ptr := torch.Unsqueeze(t.t, dim)
	return New(ptr)
}

func (t *Tensor) Squeeze(dim int64) *Tensor {
	ptr := torch.Squeeze(t.t, dim)
	return New(ptr)
}

func (t *Tensor) Flatten(startDim, endDim int64) *Tensor {
	ptr := torch.Flatten(t.t, startDim, endDim)
	return New(ptr)
}

func (t *Tensor) Contiguous() *Tensor {
	ptr := torch.Contiguous(t.t)
	return New(ptr)
}

func (t *Tensor) Expand(sizes ...int64) *Tensor {
	ptr := torch.Expand(t.t, sizes)
	return New(ptr)
}
