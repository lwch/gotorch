package torch

import (
	"github.com/lwch/gotorch/consts"
)

// #include "loss.h"
import "C"

func NewMseLoss(pred, target Tensor, reduction consts.Reduction) Tensor {
	var err *C.char
	ptr := C.new_mse_loss(&err, C.tensor(pred), C.tensor(target), C.int64_t(reduction))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func NewCrossEntropyLoss(pred, target, weight Tensor, reduction consts.Reduction, ignoreIdx int, labelSmoothing float64) Tensor {
	var err *C.char
	ptr := C.new_cross_entropy_loss(&err, C.tensor(pred), C.tensor(target), C.tensor(weight), C.int64_t(reduction), C.int64_t(ignoreIdx), C.double(labelSmoothing))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}
