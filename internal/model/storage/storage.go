package storage

import "archive/zip"

type Storage interface {
	New(size int, file *zip.File) (Storage, error)
	SetRequiresGrad(requiresGrad bool)
	Get() interface{}
}

type base struct {
	requiresGrad bool
}

func (b *base) SetRequiresGrad(requiresGrad bool) {
	b.requiresGrad = requiresGrad
}
