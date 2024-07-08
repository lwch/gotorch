package optimizer

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type AdamW struct {
	lr          float64
	weightDecay float64
	beta1       float64
	beta2       float64
	eps         float64
	amsgrad     bool
	optm        *torch.Optimizer
}

type AdamWOpt func(*AdamW)

func WithAdamWLr(lr float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.lr = lr
	}
}

func WithAdamWWeightDecay(weightDecay float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.weightDecay = weightDecay
	}
}

func WithAdamWBeta1(beta1 float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.beta1 = beta1
	}
}

func WithAdamWBeta2(beta2 float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.beta2 = beta2
	}
}

func WithAdamWEps(eps float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.eps = eps
	}
}

// NewAdamW 创建adamw优化器
//
//	lr默认为1e-3
//	weightDecay默认为1e-2
//	beta1默认为0.9
//	beta2默认为0.999
//	eps默认为1e-8
//	amsgrad默认为false
func NewAdamW(params []*tensor.Tensor, opts ...AdamWOpt) Optimizer {
	list := make([]torch.Tensor, len(params))
	for i, t := range params {
		list[i] = t.Tensor()
	}
	var adamW AdamW
	adamW.lr = 1e-3
	adamW.weightDecay = 1e-2
	adamW.beta1 = 0.9
	adamW.beta2 = 0.999
	adamW.eps = 1e-8
	for _, opt := range opts {
		opt(&adamW)
	}
	adamW.optm = torch.NewAdamWOptimizer(list, adamW.lr, adamW.beta1, adamW.beta2, adamW.eps, adamW.weightDecay, adamW.amsgrad)
	return &adamW
}

func (optm *AdamW) GetName() string {
	return "AdamW"
}

func (optm *AdamW) Step() {
	optm.optm.Step()
}

func (optm *AdamW) GetLr() float64 {
	return optm.optm.GetLr()
}

func (optm *AdamW) SetLr(lr float64) {
	optm.optm.SetLr(lr)
}

func (optm *AdamW) GetState() [][]*tensor.Tensor {
	state := optm.optm.GetState()
	ret := make([][]*tensor.Tensor, state.Size())
	for i := range ret {
		tensors := state.Get(i)
		ret[i] = make([]*tensor.Tensor, len(tensors))
		for j, t := range tensors {
			ret[i][j] = tensor.New(t)
		}
	}
	return ret
}

func (optm *AdamW) SetState(values [][]*tensor.Tensor) {
	state := optm.optm.GetState()
	for i, values := range values {
		tmp := make([]torch.Tensor, len(values))
		for j, t := range values {
			tmp[j] = t.Tensor()
		}
		state.Set(i, tmp)
	}
}
