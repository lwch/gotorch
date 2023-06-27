package torch

import (
	"github.com/lwch/gotorch/consts"
)

// #include "loss.h"
import "C"

func NewMseLoss(pred, target *Tensor, reduction consts.Reduction) *Tensor {
	var err *C.char
	ptr := C.new_mse_loss(&err, pred.data, target.data, C.int64_t(reduction))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func NewCrossEntropyLoss(pred, target, weight *Tensor, reduction consts.Reduction, ignoreIdx int, labelSmoothing float64) *Tensor {
	var err *C.char
	var ptrWeight C.tensor
	if weight != nil {
		ptrWeight = weight.data
	}
	ptr := C.new_cross_entropy_loss(&err, pred.data, target.data, ptrWeight, C.int64_t(reduction), C.int64_t(ignoreIdx), C.double(labelSmoothing))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}
