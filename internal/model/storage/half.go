package storage

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"unsafe"
)

type Half struct {
	base
	data []float32
}

var _ Storage = &Half{}

func (*Half) New(size int, file *zip.File) (Storage, error) {
	fs, err := file.Open()
	if err != nil {
		return nil, fmt.Errorf("Half.New: can not open file %s: %v", file.Name, err)
	}
	defer fs.Close()
	var ret Half
	ret.data = make([]float32, size)
	for i := 0; i < size; i++ {
		var u uint16
		err = binary.Read(fs, binary.LittleEndian, &u)
		if err != nil {
			return nil, fmt.Errorf("Half.New: can not read file %s: %v", file.Name, err)
		}
		ret.data[i] = u16toFloat(u)
	}
	return &ret, nil
}

func (f *Half) Get() interface{} {
	return f.data
}

// http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
func u16toFloat(u16 uint16) float32 {
	n := hfMantissaTable[hfOffsetTable[u16>>10]+uint32(u16&0x3ff)] + hfExponentTable[u16>>10]
	return *(*float32)(unsafe.Pointer(&n))
}

var hfMantissaTable [2048]uint32
var hfExponentTable [64]uint32
var hfOffsetTable [64]uint32

func init() {
	initMantissaTable()
	initExponentTable()
	initOffsetTable()
}

func initMantissaTable() {
	hfMantissaTable[0] = 0
	for i := 1; i <= 1023; i++ {
		hfMantissaTable[i] = convertHFMantissa(i)
	}
	for i := 1024; i <= 2047; i++ {
		hfMantissaTable[i] = 0x38000000 + uint32(i-1024)<<13
	}
}

func initExponentTable() {
	hfExponentTable[0] = 0
	hfExponentTable[31] = 0x47800000
	hfExponentTable[32] = 0x80000000
	hfExponentTable[63] = 0xC7800000
	for i := 1; i <= 30; i++ {
		hfExponentTable[i] = uint32(i) << 23
	}
	for i := 33; i <= 62; i++ {
		hfExponentTable[i] = 0x80000000 + uint32(i-32)<<23
	}
}

func initOffsetTable() {
	hfOffsetTable[0] = 0
	hfOffsetTable[32] = 0
	for i := 1; i <= 31; i++ {
		hfOffsetTable[i] = 1024
	}
	for i := 32; i <= 63; i++ {
		hfOffsetTable[i] = 1024
	}
}

func convertHFMantissa(i int) uint32 {
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
