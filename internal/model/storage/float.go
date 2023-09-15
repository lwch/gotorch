package storage

type Float struct {
	base
}

var _ Storage = &Float{}

func (f *Float) New(size int, location string) (Storage, error) {
	return &Float{}, nil
}
