package storage

import (
	"archive/zip"
	"fmt"
	"io"
	"sync"
)

type Byte struct {
	base
	data []byte
}

var _ Storage = &Byte{}

func (*Byte) New(wg *sync.WaitGroup, size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Byte.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret Byte
	ret.data = make([]byte, size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err = io.ReadFull(fs, ret.data)
		if err != nil {
			panic(fmt.Errorf("Byte.New: can not read file %s: %v", file.Name, err))
		}
	}()
	return &ret, nil
}

func (b *Byte) Get() interface{} {
	return b.data
}

func (*Byte) Type() StorageType {
	return TypeByte
}
