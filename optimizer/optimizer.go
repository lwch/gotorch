package optimizer

import "github.com/lwch/gotorch/tensor"

type Optimizer interface {
	Step()
	GetLr() float64
	SetLr(float64)
	State() [][]*tensor.Tensor
}
