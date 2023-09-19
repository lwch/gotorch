package half

import "unsafe"

// Decode http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
func Decode(u16 uint16) float32 {
	n := mantissaTable[offsetTable[u16>>10]+uint32(u16&0x3ff)] + exponentTable[u16>>10]
	return *(*float32)(unsafe.Pointer(&n))
}

var mantissaTable [2048]uint32
var exponentTable [64]uint32
var offsetTable [64]uint32

func init() {
	initMantissaTable()
	initExponentTable()
	initOffsetTable()
}

func initMantissaTable() {
	mantissaTable[0] = 0
	for i := 1; i <= 1023; i++ {
		mantissaTable[i] = convertMantissa(i)
	}
	for i := 1024; i <= 2047; i++ {
		mantissaTable[i] = 0x38000000 + uint32(i-1024)<<13
	}
}

func initExponentTable() {
	exponentTable[0] = 0
	exponentTable[31] = 0x47800000
	exponentTable[32] = 0x80000000
	exponentTable[63] = 0xC7800000
	for i := 1; i <= 30; i++ {
		exponentTable[i] = uint32(i) << 23
	}
	for i := 33; i <= 62; i++ {
		exponentTable[i] = 0x80000000 + uint32(i-32)<<23
	}
}

func initOffsetTable() {
	offsetTable[0] = 0
	offsetTable[32] = 0
	for i := 1; i <= 31; i++ {
		offsetTable[i] = 1024
	}
	for i := 32; i <= 63; i++ {
		offsetTable[i] = 1024
	}
}

func convertMantissa(i int) uint32 {
	m := uint32(i) << 13
	e := uint32(0)
	for m&0x00800000 == 0 {
		e -= 0x00800000
		m <<= 1
	}
	m &= ^uint32(0x00800000)
	e += 0x38800000
	return m | e
}
