package torch

// #include <stdint.h>
import "C"

func cints(arr []int) (*C.int64_t, C.size_t) {
	ret := make([]C.int64_t, len(arr))
	for i, v := range arr {
		ret[i] = C.int64_t(v)
	}
	return &ret[0], C.size_t(len(arr))
}
