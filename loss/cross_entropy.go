package loss

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type CrossEntropy struct {
	ignoreIdx      int
	labelSmoothing float64
	reduction      consts.Reduction
	t              *torch.Tensor
}

type CrossEntropyOpt func(*CrossEntropy)

func WithCrossEntropyReduction(reduction consts.Reduction) CrossEntropyOpt {
	return func(loss *CrossEntropy) {
		loss.reduction = reduction
	}
}

func WithCrossEntropyIgnoreIdx(idx int) CrossEntropyOpt {
	return func(loss *CrossEntropy) {
		loss.ignoreIdx = idx
	}
}

func WithCrossEntropyLabelSmoothing(smoothing float64) CrossEntropyOpt {
	return func(loss *CrossEntropy) {
		loss.labelSmoothing = smoothing
	}
}

// NewCrossEntropy 创建CrossEntropy损失函数
//
//	reduction默认为mean
//	ignoreIdx默认为-100
//	labelSmoothing默认为0
func NewCrossEntropy(pred, target *tensor.Tensor, opts ...CrossEntropyOpt) Loss {
	var ret CrossEntropy
	for _, opt := range opts {
		opt(&ret)
	}
	ret.t = torch.NewCrossEntropyLoss(pred.Tensor(), target.Tensor(),
		ret.reduction, ret.ignoreIdx, ret.labelSmoothing)
	if pred.Storage() != nil {
		pred.Storage().Put(ret.t)
	} else if target.Storage() != nil {
		target.Storage().Put(ret.t)
	}
	return &ret
}

func (loss *CrossEntropy) Backward() {
	loss.t.Backward()
}

func (loss *CrossEntropy) Value() float64 {
	switch loss.t.ScalarType() {
	case consts.KFloat:
		return float64(loss.t.Float32Value()[0])
	case consts.KDouble:
		return loss.t.Float64Value()[0]
	default:
		panic("not implemented")
	}
}
