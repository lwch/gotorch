package nn

import (
	"github.com/lwch/gotorch/internal/torch"
)

type LayerNorm struct {
	module
}

func NewLayerNorm(name string, shapes ...int64) *LayerNorm {
	return &LayerNorm{
		module{
			name: name,
			m:    torch.NewLayerNorm(shapes),
		},
	}
}
