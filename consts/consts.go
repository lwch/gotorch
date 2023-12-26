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
	KQInt8
	KQUInt8
	KQInt32
	KBFloat16
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
	case KQInt8:
		return "qint8"
	case KQUInt8:
		return "quint8"
	case KQInt32:
		return "qint32"
	case KBFloat16:
		return "bfloat16"
	default:
		return "unknown scalar type"
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

func (t DeviceType) String() string {
	switch t {
	case KCPU:
		return "cpu"
	case KCUDA:
		return "cuda"
	case KMKLDNN:
		return "mkldnn"
	case KOPENGL:
		return "opengl"
	case KOPENCL:
		return "opencl"
	case KIDEEP:
		return "ideep"
	case KHIP:
		return "hip"
	case KFPGA:
		return "fpga"
	case KORT:
		return "ort"
	case KXLA:
		return "xla"
	case KVulkan:
		return "vulkan"
	case KMetal:
		return "metal"
	case KXPU:
		return "xpu"
	case KMPS:
		return "mps"
	case KMeta:
		return "meta"
	case KHPU:
		return "hpu"
	case KVE:
		return "ve"
	case KLazy:
		return "lazy"
	case KIPU:
		return "ipu"
	case KMTIA:
		return "mtia"
	default:
		return "unknown device"
	}
}

type Reduction byte

const (
	ReductionNone Reduction = iota
	ReductionMean
	ReductionSum
	ReductionEnd
)
