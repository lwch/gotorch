package optimizer

import (
	"encoding/binary"
	"io"

	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type adamOptions struct {
	Lr          float64
	WeightDecay float64
	Beta1       float64
	Beta2       float64
	Eps         float64
}

func (opts *adamOptions) WriteTo(w io.Writer) (int64, error) {
	return 5 * 8, binary.Write(w, binary.LittleEndian, opts)
}

func (opts *adamOptions) ReadFrom(r io.Reader) (int64, error) {
	return 5 * 8, binary.Read(r, binary.LittleEndian, opts)
}

type Adam struct {
	options adamOptions
	optm    *torch.Optimizer
}

type AdamOpt func(*Adam)

func WithAdamLr(lr float64) AdamOpt {
	return func(adam *Adam) {
		adam.options.Lr = lr
	}
}

func WithAdamWeightDecay(weightDecay float64) AdamOpt {
	return func(adam *Adam) {
		adam.options.WeightDecay = weightDecay
	}
}

func WithAdamBeta1(beta1 float64) AdamOpt {
	return func(adam *Adam) {
		adam.options.Beta1 = beta1
	}
}

func WithAdamBeta2(beta2 float64) AdamOpt {
	return func(adam *Adam) {
		adam.options.Beta2 = beta2
	}
}

func WithAdamEps(eps float64) AdamOpt {
	return func(adam *Adam) {
		adam.options.Eps = eps
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
	adam.options.Lr = 1e-3
	adam.options.WeightDecay = 0
	adam.options.Beta1 = 0.9
	adam.options.Beta2 = 0.999
	adam.options.Eps = 1e-8
	for _, opt := range opts {
		opt(&adam)
	}
	adam.optm = torch.NewAdamOptimizer(list,
		adam.options.Lr,
		adam.options.Beta1, adam.options.Beta2,
		adam.options.Eps, adam.options.WeightDecay)
	return &adam
}

func (optm *Adam) GetName() string {
	return "Adam"
}

func (optm *Adam) Step() {
	optm.optm.Step()
}

func (optm *Adam) ZeroGrad() {
	optm.optm.ZeroGrad()
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

func (optm *Adam) GetOptions() Options {
	return &optm.options
}
