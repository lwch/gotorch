package tensor

import (
	"github.com/lwch/gotorch/consts"
)

// #include "loss.h"
import "C"

func NewMseLoss(pred, target *Tensor, reduction consts.Reduction) *Tensor {
	ptr := C.new_mse_loss(pred.data, target.data, C.int64_t(reduction))
	return &Tensor{data: ptr}
}

func NewCrossEntropyLoss(pred, target *Tensor, reduction consts.Reduction, ignoreIdx int, labelSmoothing float64) *Tensor {
	ptr := C.new_cross_entropy_loss(pred.data, target.data, C.int64_t(reduction), C.int64_t(ignoreIdx), C.double(labelSmoothing))
	return &Tensor{data: ptr}
}
