package storage

type BFloat16 struct {
	base
}

var _ Storage = &BFloat16{}

func (f *BFloat16) New(size int, location string) (Storage, error) {
	return &BFloat16{}, nil
}
