package loss

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type crossEntropyConfig struct {
	weight         *tensor.Tensor
	ignoreIdx      int
	labelSmoothing float64
	reduction      consts.Reduction
}

type CrossEntropyOpt func(*crossEntropyConfig)

func WithCrossEntropyWeight(w *tensor.Tensor) CrossEntropyOpt {
	return func(loss *crossEntropyConfig) {
		loss.weight = w
	}
}

func WithCrossEntropyReduction(reduction consts.Reduction) CrossEntropyOpt {
	return func(loss *crossEntropyConfig) {
		loss.reduction = reduction
	}
}

func WithCrossEntropyIgnoreIdx(idx int) CrossEntropyOpt {
	return func(loss *crossEntropyConfig) {
		loss.ignoreIdx = idx
	}
}

func WithCrossEntropyLabelSmoothing(smoothing float64) CrossEntropyOpt {
	return func(loss *crossEntropyConfig) {
		loss.labelSmoothing = smoothing
	}
}

// NewCrossEntropy 创建CrossEntropy损失函数
//
//	reduction默认为mean
//	ignoreIdx默认为-100
//	labelSmoothing默认为0
func NewCrossEntropy(pred, target *tensor.Tensor, opts ...CrossEntropyOpt) *tensor.Tensor {
	var cfg crossEntropyConfig
	cfg.reduction = consts.ReductionMean
	cfg.ignoreIdx = -100
	cfg.labelSmoothing = 0
	for _, opt := range opts {
		opt(&cfg)
	}
	var weight torch.Tensor
	if cfg.weight != nil {
		weight = cfg.weight.Tensor()
	}
	ptr := torch.NewCrossEntropyLoss(pred.Tensor(), target.Tensor(),
		weight, cfg.reduction, cfg.ignoreIdx, cfg.labelSmoothing)
	return tensor.New(ptr)
}
