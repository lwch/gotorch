package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Half struct {
	base
	data []uint16
}

var _ Storage = &Half{}

func (*Half) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Half.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret Half
	ret.data = make([]uint16, size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Half.New: can not read file %s: %v", file.Name, err))
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
