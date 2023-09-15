package model

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"

	"github.com/lwch/gotorch/internal/model/storage"
	"github.com/lwch/gotorch/internal/model/torch"
	"github.com/nlpodyssey/gopickle/pickle"
	"github.com/nlpodyssey/gopickle/types"
)

type findClassFunc func(module, name string) (interface{}, error)

type Model struct {
	storages map[string]storage.Storage
	files    map[string]*zip.File
	params   map[string]storage.Storage
}

func Load(dir string) (*Model, error) {
	f, err := os.Open(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	zr, err := zip.NewReader(f, fi.Size())
	if err != nil {
		return nil, err
	}
	data, err := getDataPkl(zr)
	if err != nil {
		return nil, err
	}
	defer data.Close()
	pkl := pickle.NewUnpickler(data)
	var m Model
	m.storages = make(map[string]storage.Storage)
	m.files = files(zr)
	pkl.FindClass = m.buildFindClass(pkl.FindClass)
	pkl.PersistentLoad = m.persistentLoad
	params, err := pkl.Load()
	if err != nil {
		return nil, err
	}
	err = m.loadParams(params)
	if err != nil {
		return nil, err
	}
	return &m, nil
}

func getDataPkl(r *zip.Reader) (io.ReadCloser, error) {
	for _, f := range r.File {
		if filepath.Base(f.Name) == "data.pkl" {
			return f.Open()
		}
	}
	return nil, errors.New("data.pkl not found")
}

func files(r *zip.Reader) map[string]*zip.File {
	result := make(map[string]*zip.File)
	for _, file := range r.File {
		_, recordName := path.Split(file.Name)
		result[recordName] = file
	}
	return result
}

func (m *Model) buildFindClass(cb findClassFunc) findClassFunc {
	return func(module, name string) (interface{}, error) {
		switch module + "." + name {
		case "torch._utils._rebuild_tensor_v2":
			return &torch.RebuildTensorV2{}, nil
		case "torch._utils._rebuild_parameter":
			return &torch.RebuildParameter{}, nil
		case "torch.HalfStorage": // half, float16
			return &storage.Half{}, nil
		case "torch.BFloat16Storage": // bfloat16
			return &storage.BFloat16{}, nil
		case "torch.FloatStorage": // float32, float
			return &storage.Float{}, nil
		case "torch.DoubleStorage": // float64, double
			return &storage.Double{}, nil
		case "torch.ByteStorage": // uint8
			return &storage.Byte{}, nil
		case "torch.CharStorage": // int8
			return &storage.Char{}, nil
		case "torch.ShortStorage": // int16, short
			return &storage.Short{}, nil
		case "torch.IntStorage": // int32, int
			return &storage.Int{}, nil
		case "torch.LongStorage": // int64, long
			return &storage.Long{}, nil
		default:
			if cb == nil {
				return nil, fmt.Errorf("class not found: %s %s", module, name)
			}
			return cb(module, name)
		}
	}
}

func (m *Model) persistentLoad(id interface{}) (interface{}, error) {
	tuple, tupleOk := id.(*types.Tuple)
	if !tupleOk || tuple.Len() == 0 {
		return nil, fmt.Errorf("PersistentLoad: non-empty tuple expected, got %#v", id)
	}
	typename, typenameOk := tuple.Get(0).(string)
	if !typenameOk {
		return nil, fmt.Errorf("PersistentLoad: cannot get typename")
	}
	if typename != "storage" {
		return nil, fmt.Errorf("unknown typename for PersistentLoad, expected 'storage' but got '%s'", typename)
	}
	if tuple.Len() < 5 {
		return nil, fmt.Errorf("PersistentLoad: unexpected storage data length")
	}
	storageType, storageTypeOk := tuple.Get(1).(storage.Storage)
	key, keyOk := tuple.Get(2).(string)
	size, sizeOk := tuple.Get(4).(int)
	if !storageTypeOk || !keyOk || !sizeOk {
		return nil, fmt.Errorf("PersistentLoad: unexpected data types")
	}

	storage, ok := m.storages[key]
	if !ok {
		file, ok := m.files[key]
		if !ok {
			return nil, fmt.Errorf("PersistentLoad: file not found: %s", key)
		}
		var err error
		storage, err = storageType.New(size, file)
		if err != nil {
			return nil, err
		}
		m.storages[key] = storage
	}
	return storage, nil
}

func (m *Model) loadParams(params interface{}) error {
	switch params.(type) {
	case *types.OrderedDict:
		return m.loadParamsOrderedDict(params)
	case *types.Dict:
		return m.loadParamsDict(params)
	default:
		return fmt.Errorf("loadParams: unexpected data type: %#v", params)
	}
}

func (m *Model) loadParamsOrderedDict(params interface{}) error {
	dict := params.(*types.OrderedDict)
	m.params = make(map[string]storage.Storage)
	for _, entry := range dict.Map {
		key, keyOk := entry.Key.(string)
		value, valueOk := entry.Value.(storage.Storage)
		if !keyOk || !valueOk {
			return fmt.Errorf("loadParamsOrderedDict: unexpected data types, key: %#v, value: %#v", keyOk, valueOk)
		}
		m.params[key] = value
	}
	return nil
}

func (m *Model) loadParamsDict(params interface{}) error {
	dict := params.(*types.Dict)
	m.params = make(map[string]storage.Storage)
	for _, entry := range *dict {
		key, keyOk := entry.Key.(string)
		value, valueOk := entry.Value.(storage.Storage)
		if !keyOk || !valueOk {
			return fmt.Errorf("loadParamsDict: unexpected data types, key: %#v, value: %#v", keyOk, valueOk)
		}
		m.params[key] = value
	}
	return nil
}

func (m *Model) Params() map[string]storage.Storage {
	return m.params
}
