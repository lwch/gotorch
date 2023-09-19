package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
	"unsafe"
)

type BFloat16 struct {
	base
	data []float32
}

var _ Storage = &BFloat16{}

func (*BFloat16) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Half.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret BFloat16
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
			ret.data[i] = u16toBFloat(u)
		}
	}()
	return &ret, nil
}

func (f *BFloat16) Get() interface{} {
	return f.data
}

func (*BFloat16) Type() StorageType {
	return TypeBFloat16
}

func u16toBFloat(u16 uint16) float32 {
	n := uint32(u16) << 16
	return *(*float32)(unsafe.Pointer(&n))
}
