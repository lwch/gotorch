package nn

import (
	"github.com/lwch/gotorch/internal/torch"
)

type LayerNorm struct {
	module
}

func NewLayerNorm(shapes ...int64) *LayerNorm {
	return &LayerNorm{
		module{torch.NewLayerNorm(shapes)},
	}
}
