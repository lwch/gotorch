package optimizer

import (
	"encoding/binary"
	"io"

	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type adamWOptions struct {
	Lr          float64
	WeightDecay float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	Amsgrad     bool
}

func (opts *adamWOptions) WriteTo(w io.Writer) (int64, error) {
	return 5*8 + 1, binary.Write(w, binary.LittleEndian, opts)
}

func (opts *adamWOptions) ReadFrom(r io.Reader) (int64, error) {
	return 5*8 + 1, binary.Read(r, binary.LittleEndian, opts)
}

type AdamW struct {
	options adamWOptions
	optm    *torch.Optimizer
}

type AdamWOpt func(*AdamW)

func WithAdamWLr(lr float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.options.Lr = lr
	}
}

func WithAdamWWeightDecay(weightDecay float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.options.WeightDecay = weightDecay
	}
}

func WithAdamWBeta1(beta1 float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.options.Beta1 = beta1
	}
}

func WithAdamWBeta2(beta2 float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.options.Beta2 = beta2
	}
}

func WithAdamWEps(eps float64) AdamWOpt {
	return func(adam *AdamW) {
		adam.options.Eps = eps
	}
}

func WithAdamWAmsgrad(amsgrad bool) AdamWOpt {
	return func(adam *AdamW) {
		adam.options.Amsgrad = amsgrad
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
	adamW.options.Lr = 1e-3
	adamW.options.WeightDecay = 1e-2
	adamW.options.Beta1 = 0.9
	adamW.options.Beta2 = 0.999
	adamW.options.Eps = 1e-8
	adamW.options.Amsgrad = false
	for _, opt := range opts {
		opt(&adamW)
	}
	adamW.optm = torch.NewAdamWOptimizer(list,
		adamW.options.Lr,
		adamW.options.Beta1, adamW.options.Beta2,
		adamW.options.Eps,
		adamW.options.WeightDecay,
		adamW.options.Amsgrad)
	return &adamW
}

func (optm *AdamW) GetName() string {
	return "AdamW"
}

func (optm *AdamW) Step() {
	optm.optm.Step()
}

func (optm *AdamW) ZeroGrad() {
	optm.optm.ZeroGrad()
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

func (optm *AdamW) GetOptions() Options {
	return &optm.options
}
