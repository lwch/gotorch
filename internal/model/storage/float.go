package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"sync"
)

type Float struct {
	base
	data []float32
}

var _ Storage = &Float{}

func (*Float) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Float.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret Float
	ret.data = make([]float32, size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		err = binary.Read(fs, binary.LittleEndian, ret.data)
		if err != nil {
			panic(fmt.Errorf("Float.New: can not read file %s: %v", file.Name, err))
		}
	}()
	return &ret, nil
}

func (f *Float) Get() interface{} {
	return f.data
}

func (*Float) Type() StorageType {
	return TypeFloat
}
