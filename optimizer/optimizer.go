package optimizer

import "github.com/lwch/gotorch/tensor"

type Optimizer interface {
	GetName() string
	Step()
	GetLr() float64
	SetLr(float64)
	GetState() [][]*tensor.Tensor
	SetState([][]*tensor.Tensor)
}
