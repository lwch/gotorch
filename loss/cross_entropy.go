package loss

import (
	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

type CrossEntropy struct {
	weight         *tensor.Tensor
	ignoreIdx      int
	labelSmoothing float64
	reduction      consts.Reduction
	t              *tensor.Tensor
}

type CrossEntropyOpt func(*CrossEntropy)

func WithCrossEntropyWeight(w *tensor.Tensor) CrossEntropyOpt {
	return func(loss *CrossEntropy) {
		loss.weight = w
	}
}

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
	ret.reduction = consts.ReductionMean
	ret.ignoreIdx = -100
	ret.labelSmoothing = 0
	for _, opt := range opts {
		opt(&ret)
	}
	var weight torch.Tensor
	if ret.weight != nil {
		weight = ret.weight.Tensor()
	}
	ptr := torch.NewCrossEntropyLoss(pred.Tensor(), target.Tensor(),
		weight, ret.reduction, ret.ignoreIdx, ret.labelSmoothing)
	ret.t = tensor.New(ptr, "CrossEntropyLoss")
	return &ret
}

func (loss *CrossEntropy) Backward() {
	loss.t.Backward()
}

func (loss *CrossEntropy) BackwardRetained() {
	loss.t.BackwardRetained()
}

func (loss *CrossEntropy) Value() float64 {
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
