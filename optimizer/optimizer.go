package optimizer

import (
	"io"

	"github.com/lwch/gotorch/tensor"
)

type Options interface {
	io.WriterTo
	io.ReaderFrom
}

type Optimizer interface {
	GetName() string
	Step()
	ZeroGrad()
	GetLr() float64
	SetLr(float64)
	GetState() [][]*tensor.Tensor
	SetState([][]*tensor.Tensor)
	GetOptions() Options
}
