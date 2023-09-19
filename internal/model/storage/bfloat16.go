package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type BFloat16 struct {
	base
	data []uint16
}

var _ Storage = &BFloat16{}

func (*BFloat16) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("BFloat16.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret BFloat16
	ret.data = make([]uint16, size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("BFloat16.New: can not read file %s: %v", file.Name, err))
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
