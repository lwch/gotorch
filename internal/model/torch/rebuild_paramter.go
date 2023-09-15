package torch

import (
	"fmt"

	"github.com/lwch/gotorch/internal/model/storage"
	"github.com/nlpodyssey/gopickle/types"
)

type RebuildParameter struct {
}

var _ types.Callable = &RebuildParameter{}

func (r *RebuildParameter) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("RebuildParameter unexpected 3 args, got %d: %#v", len(args), args)
	}

	storage, storageOk := args[0].(storage.Storage)
	requiresGrad, requiresGradOk := args[1].(bool)
	if !storageOk || !requiresGradOk {
		return nil, fmt.Errorf("RebuildParameter unexpected args: %#v", args)
	}

	storage.SetRequiresGrad(requiresGrad)

	return storage, nil
}
