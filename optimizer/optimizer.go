package optimizer

import "github.com/lwch/gotorch/tensor"

type Optimizer interface {
	Step([]*tensor.Tensor)
	GetLr() float64
	SetLr(float64)
}
