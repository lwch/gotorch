package storage

type Storage interface {
	New(size int, location string) (Storage, error)
	SetRequiresGrad(requiresGrad bool)
}

type base struct {
	requiresGrad bool
}

func (b *base) SetRequiresGrad(requiresGrad bool) {
	b.requiresGrad = requiresGrad
}
