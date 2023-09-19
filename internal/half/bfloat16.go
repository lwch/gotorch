package half

import "unsafe"

func EncodeBFloat16(f float32) uint16 {
	n := *(*uint32)(unsafe.Pointer(&f))
	return uint16(n >> 16)
}

func DecodeBFloat16(u16 uint16) float32 {
	n := uint32(u16) << 16
	return *(*float32)(unsafe.Pointer(&n))
}
