package init

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/tensor"
)

func KaimingUniform(t *tensor.Tensor, a float64) {
	torch.KaimingUniform(t.Tensor(), a)
}

func XaiverUniform(t *tensor.Tensor, gain float64) {
	torch.XaiverUniform(t.Tensor(), gain)
}

func Normal(t *tensor.Tensor, mean, std float64) {
	torch.Normal(t.Tensor(), mean, std)
}
