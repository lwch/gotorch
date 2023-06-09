package tensor

import (
	"github.com/lwch/gotorch/internal/torch"
	"github.com/lwch/gotorch/mmgr"
)

type Tensor struct {
	s *mmgr.Storage
	t *torch.Tensor
}

func (t *Tensor) Storage() *mmgr.Storage {
	return t.s
}

func (t *Tensor) Tensor() *torch.Tensor {
	return t.t
}
