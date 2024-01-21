package loss

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type mseConfig struct {
	reduction consts.Reduction
}

type MseOpt func(*mseConfig)

func WithMseReduction(reduction consts.Reduction) MseOpt {
	return func(loss *mseConfig) {
		loss.reduction = reduction
	}
}

// NewMse reduction默认为Mean
func NewMse(pred, target *tensor.Tensor, opts ...MseOpt) *tensor.Tensor {
	var ret mseConfig
	ret.reduction = consts.ReductionMean
	for _, opt := range opts {
		opt(&ret)
	}
	ptr := torch.NewMseLoss(pred.Tensor(), target.Tensor(), ret.reduction)
	return tensor.New(ptr)
}
