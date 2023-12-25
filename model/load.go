package model

import (
	"sync"

	"github.com/lwch/gotorch/internal/model"
	"github.com/lwch/gotorch/internal/model/storage"
	"github.com/lwch/gotorch/tensor"
)

type Model struct {
	params map[string]*tensor.Tensor
}

func Load(dir string) (*Model, error) {
	m, err := model.Load(dir)
	if err != nil {
		return nil, err
	}
	var mu sync.Mutex
	params := make(map[string]*tensor.Tensor)
	var wg sync.WaitGroup
	wg.Add(len(m.Params()))
	for k, v := range m.Params() {
		go func(k string, v storage.Storage) {
			defer wg.Done()
			t := buildTensor(v)
			mu.Lock()
			params[k] = t
			mu.Unlock()
		}(k, v)
	}
	wg.Wait()
	return &Model{params: params}, nil
}

func buildTensor(t storage.Storage) *tensor.Tensor {
	switch t.Type() {
	// float to bfloat16 tensor
	case storage.TypeBFloat16:
		return tensor.FromBFloat16Raw(t.Get().([]uint16),
			tensor.WithShapes(t.GetShape()...))
	// float to half tensor
	case storage.TypeHalf:
		return tensor.FromHalfRaw(t.Get().([]uint16),
			tensor.WithShapes(t.GetShape()...))
	// float to float32 tensor
	case storage.TypeFloat:
		return tensor.FromFloat32(t.Get().([]float32),
			tensor.WithShapes(t.GetShape()...))
	// double to float64 tensor
	case storage.TypeDouble:
		return tensor.FromFloat64(t.Get().([]float64),
			tensor.WithShapes(t.GetShape()...))
	// byte to uint8 tensor
	case storage.TypeByte:
		return tensor.FromUint8(t.Get().([]byte),
			tensor.WithShapes(t.GetShape()...))
	// char to int8 tensor
	case storage.TypeChar:
		return tensor.FromInt8(t.Get().([]int8),
			tensor.WithShapes(t.GetShape()...))
	// short to int16 tensor
	case storage.TypeShort:
		return tensor.FromInt16(t.Get().([]int16),
			tensor.WithShapes(t.GetShape()...))
	// int to int32 tensor
	case storage.TypeInt:
		return tensor.FromInt32(t.Get().([]int32),
			tensor.WithShapes(t.GetShape()...))
	// long to int64 tensor
	case storage.TypeLong:
		return tensor.FromInt64(t.Get().([]int64),
			tensor.WithShapes(t.GetShape()...))
	default:
		panic("not supported")
	}
}

func (m *Model) Get(name string) *tensor.Tensor {
	return m.params[name]
}

func (m *Model) Params() map[string]*tensor.Tensor {
	return m.params
}
