package nn

import (
	"github.com/lwch/gotorch/internal/torch"
)

type Linear struct {
	module
}

func NewLinear(name string, inFeatures, outFeatures int64) *Linear {
	return &Linear{
		module{
			name: name,
			m:    torch.NewLinear(inFeatures, outFeatures),
		},
	}
}
