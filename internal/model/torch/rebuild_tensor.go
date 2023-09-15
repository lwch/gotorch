package torch

import (
	"fmt"

	"github.com/lwch/gotorch/internal/model/storage"
	"github.com/nlpodyssey/gopickle/types"
)

// https://github.com/pytorch/pytorch/blob/main/torch/_utils.py

type RebuildTensorV2 struct{}

var _ types.Callable = &RebuildTensorV2{}

func (r *RebuildTensorV2) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 6 {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	storage, storageOk := args[0].(storage.Storage)
	size, sizeOk := args[2].(*types.Tuple)
	requiresGrad, requiresGradOk := args[4].(bool)
	if !storageOk || !sizeOk || !requiresGradOk {
		return nil, fmt.Errorf("RebuildTensorV2 unexpected args: %#v", args)
	}

	shape, err := tupleToInt64Slice(size)
	if err != nil {
		return nil, fmt.Errorf("RebuildTensorV2: %v", err)
	}

	storage.SetShape(shape)
	storage.SetRequiresGrad(requiresGrad)

	return storage, nil
}

func tupleToInt64Slice(tuple *types.Tuple) ([]int64, error) {
	length := tuple.Len()
	slice := make([]int64, length)
	for i := 0; i < length; i++ {
		value, ok := tuple.Get(i).(int)
		if !ok {
			// return nil, fmt.Errorf("tuple of ints expected. Got %#v", tuple)
			fmt.Printf("WARNING: tuple of ints expected. Got %#v\n", tuple)
			continue
		}
		slice[i] = int64(value)
	}
	return slice, nil
}
