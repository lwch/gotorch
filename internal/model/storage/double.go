package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
)

type Double struct {
	base
	data []float64
}

var _ Storage = &Double{}

func (*Double) New(size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Double.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret Double
	ret.data = make([]float64, size)
	err = binary.Read(fs, binary.LittleEndian, ret.data)
	if err != nil {
		return nil, fmt.Errorf("Double.New: can not read file %s: %v", file.Name, err)
	}
	return &ret, nil
}

func (f *Double) Get() interface{} {
	return f.data
}

func (*Double) Type() StorageType {
	return TypeDouble
}
