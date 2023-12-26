package loss

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type Mse struct {
	reduction consts.Reduction
	t         *tensor.Tensor
}

type MseOpt func(*Mse)

func WithMseReduction(reduction consts.Reduction) MseOpt {
	return func(loss *Mse) {
		loss.reduction = reduction
	}
}

// NewMse reduction默认为Mean
func NewMse(pred, target *tensor.Tensor, opts ...MseOpt) Loss {
	var ret Mse
	ret.reduction = consts.ReductionMean
	for _, opt := range opts {
		opt(&ret)
	}
	ptr := torch.NewMseLoss(pred.Tensor(), target.Tensor(), ret.reduction)
	ret.t = tensor.New(ptr)
	return &ret
}

func (loss *Mse) Backward() {
	loss.t.Backward()
}

func (loss *Mse) BackwardRetained() {
	loss.t.BackwardRetained()
}

func (loss *Mse) Value() float64 {
	l := loss.t.ToDevice(consts.KCPU)
	switch loss.t.ScalarType() {
	case consts.KFloat:
		return float64(l.Float32Value()[0])
	case consts.KDouble:
		return l.Float64Value()[0]
	default:
		panic("not implemented")
	}
}
