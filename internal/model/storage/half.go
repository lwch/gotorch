package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"

	"github.com/lwch/gotorch/internal/half"
)

type Half struct {
	base
	data []float32
}

var _ Storage = &Half{}

func (*Half) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Half.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret Half
	ret.data = make([]float32, size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < size; i++ {
			var u uint16
			err = binary.Read(fs, binary.LittleEndian, &u)
			if err != nil {
				panic(fmt.Errorf("Half.New: can not read file %s: %v", file.Name, err))
			}
			ret.data[i] = half.Decode(u)
		}
	}()
	return &ret, nil
}

func (f *Half) Get() interface{} {
	return f.data
}

func (*Half) Type() StorageType {
	return TypeHalf
}
