package optimizer

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type Adam struct {
	lr          float64
	weightDecay float64
	beta1       float64
	beta2       float64
	eps         float64
	optm        *torch.Optimizer
}

type AdamOpt func(*Adam)

func WithAdamLr(lr float64) AdamOpt {
	return func(adam *Adam) {
		adam.lr = lr
	}
}

func WithAdamWeightDecay(weightDecay float64) AdamOpt {
	return func(adam *Adam) {
		adam.weightDecay = weightDecay
	}
}

func WithAdamBeta1(beta1 float64) AdamOpt {
	return func(adam *Adam) {
		adam.beta1 = beta1
	}
}

func WithAdamBeta2(beta2 float64) AdamOpt {
	return func(adam *Adam) {
		adam.beta2 = beta2
	}
}

func WithAdamEps(eps float64) AdamOpt {
	return func(adam *Adam) {
		adam.eps = eps
	}
}

// NewAdam 创建adam优化器
//
//	lr默认为1e-3
//	weightDecay默认为0
//	beta1默认为0.9
//	beta2默认为0.999
//	eps默认为1e-8
func NewAdam(params []*tensor.Tensor, opts ...AdamOpt) Optimizer {
	list := make([]torch.Tensor, len(params))
	for i, t := range params {
		list[i] = t.Tensor()
	}
	var adam Adam
	adam.lr = 1e-3
	adam.weightDecay = 0
	adam.beta1 = 0.9
	adam.beta2 = 0.999
	adam.eps = 1e-8
	for _, opt := range opts {
		opt(&adam)
	}
	adam.optm = torch.NewAdamOptimizer(list, adam.lr, adam.beta1, adam.beta2, adam.eps, adam.weightDecay)
	return &adam
}

func (optm *Adam) GetName() string {
	return "Adam"
}

func (optm *Adam) Step() {
	optm.optm.Step()
}

func (optm *Adam) GetLr() float64 {
	return optm.optm.GetLr()
}

func (optm *Adam) SetLr(lr float64) {
	optm.optm.SetLr(lr)
}

func (optm *Adam) GetState() [][]*tensor.Tensor {
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

func (optm *Adam) SetState(values [][]*tensor.Tensor) {
	state := optm.optm.GetState()
	for i, values := range values {
		tmp := make([]torch.Tensor, len(values))
		for j, t := range values {
			tmp[j] = t.Tensor()
		}
		state.Set(i, tmp)
	}
}
