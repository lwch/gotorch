package consts

type ScalarType byte

const (
	KUint8 ScalarType = iota
	KInt8
	KInt16
	KInt32
	KInt64
	KHalf
	KFloat
	KDouble
	KComplexHalf
	KComplexFloat
	KComplexDouble
	KBool
)

func (t ScalarType) String() string {
	switch t {
	case KUint8:
		return "uint8"
	case KInt8:
		return "int8"
	case KInt16:
		return "int16"
	case KInt32:
		return "int32"
	case KInt64:
		return "int64"
	case KHalf:
		return "half"
	case KFloat:
		return "float"
	case KDouble:
		return "double"
	case KComplexHalf:
		return "complex half"
	case KComplexFloat:
		return "complex float"
	case KComplexDouble:
		return "complex double"
	case KBool:
		return "bool"
	default:
		return "unknown"
	}
}

type DeviceType byte

const (
	KCPU DeviceType = iota
	KCUDA
	KMKLDNN
	KOPENGL
	KOPENCL
	KIDEEP
	KHIP
	KFPGA
	KORT
	KXLA
	KVulkan
	KMetal
	KXPU
	KMPS
	KMeta
	KHPU
	KVE
	KLazy
	KIPU
	KMTIA
)

type Reduction byte

const (
	ReductionNone Reduction = iota
	ReductionMean
	ReductionSum
	ReductionEnd
)
