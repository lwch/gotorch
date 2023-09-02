package nn

import (
	"github.com/lwch/gotorch/internal/torch"
)

type Linear struct {
	module
}

func NewLinear(inFeatures, outFeatures int64) *Linear {
	return &Linear{
		module{torch.NewLinear(inFeatures, outFeatures)},
	}
}
