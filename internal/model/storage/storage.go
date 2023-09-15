package storage

import "archive/zip"

type StorageType byte

const (
	TypeBFloat16 StorageType = iota // torch.bfloat16
	TypeHalf                        // torch.half, torch.float16
	TypeFloat                       // torch.float, torch.float32
	TypeDouble                      // torch.double, torch.float64
	TypeByte                        // torch.byte, torch.uint8
	TypeChar                        // torch.char, torch.int8
	TypeShort                       // torch.short, torch.int16
	TypeInt                         // torch.int, torch.int32
	TypeLong                        // torch.long, torch.int64
)

type Storage interface {
	New(size int, file *zip.File) (Storage, error)
	SetShape(shape []int64)
	GetShape() []int64
	SetRequiresGrad(requiresGrad bool)
	GetRequiresGrad() bool
	Type() StorageType
	Get() interface{}
}

type base struct {
	shape        []int64
	requiresGrad bool
}

func (b *base) SetShape(shape []int64) {
	b.shape = shape
}

func (b *base) GetShape() []int64 {
	return b.shape
}

func (b *base) SetRequiresGrad(requiresGrad bool) {
	b.requiresGrad = requiresGrad
}

func (b *base) GetRequiresGrad() bool {
	return b.requiresGrad
}
