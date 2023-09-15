package storage

import "archive/zip"

type BFloat16 struct {
	base
	data []float32
}

var _ Storage = &BFloat16{}

func (*BFloat16) New(size int, file *zip.File) (Storage, error) {
	// TODO
	return &BFloat16{}, nil
}

func (f *BFloat16) Get() interface{} {
	return f.data
}
