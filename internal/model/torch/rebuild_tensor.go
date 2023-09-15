package torch

import (
	"fmt"

	"github.com/lwch/gotorch/internal/model/storage"
	"github.com/nlpodyssey/gopickle/types"
)

type RebuildTensorV2 struct {
}

var _ types.Callable = &RebuildTensorV2{}

func (r *RebuildTensorV2) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 6 {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	storage, storageOk := args[0].(storage.Storage)

	requiresGrad, requiresGradOk := args[4].(bool)
	if !storageOk || !requiresGradOk {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	storage.SetRequiresGrad(requiresGrad)

	return storage, nil
}
