package half

import "unsafe"

// Encode http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
func Encode(f float32) uint16 {
	n := *(*uint32)(unsafe.Pointer(&f))
	return baseTable[(n>>23)&0x1ff] + uint16((n&0x007fffff)>>shiftTable[(n>>23)&0x1ff])
}

var baseTable [512]uint16
var shiftTable [512]uint16

func init() {
	var e int
	for i := 0; i < 256; i++ {
		e = i - 127
		if e < -24 { // Very small numbers map to zero
			baseTable[i|0x000] = 0x0000
			baseTable[i|0x100] = 0x8000
			shiftTable[i|0x000] = 24
			shiftTable[i|0x100] = 24
		} else if e < -14 { // Small numbers map to denorms
			baseTable[i|0x000] = (0x0400 >> (-e - 14))
			baseTable[i|0x100] = (0x0400 >> (-e - 14)) | 0x8000
			shiftTable[i|0x000] = uint16(-e - 1)
			shiftTable[i|0x100] = uint16(-e - 1)
		} else if e <= 15 { // Normal numbers just lose precision
			baseTable[i|0x000] = uint16((e + 15) << 10)
			baseTable[i|0x100] = uint16((e+15)<<10) | 0x8000
			shiftTable[i|0x000] = 13
			shiftTable[i|0x100] = 13
		} else if e < 128 { // Large numbers map to Infinity
			baseTable[i|0x000] = 0x7C00
			baseTable[i|0x100] = 0xFC00
			shiftTable[i|0x000] = 24
			shiftTable[i|0x100] = 24
		} else { // Infinity and NaN's stay Infinity and NaN's
			baseTable[i|0x000] = 0x7C00
			baseTable[i|0x100] = 0xFC00
			shiftTable[i|0x000] = 13
			shiftTable[i|0x100] = 13
		}
	}
}
