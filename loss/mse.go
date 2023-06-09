package loss

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type Mse struct {
	reduction consts.Reduction
	t         *torch.Tensor
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
	ret.t = torch.NewMseLoss(pred.Tensor(), target.Tensor(), ret.reduction)
	if pred.Storage() != nil {
		pred.Storage().Put(ret.t)
	} else if target.Storage() != nil {
		target.Storage().Put(ret.t)
	}
	return &ret
}

func (loss *Mse) Backward() {
	loss.t.Backward()
}
