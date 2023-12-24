package torch

// #include <stdint.h>
// #include "module.h"
import "C"
import (
	"math"

	"github.com/lwch/gotorch/consts"
)

type module C.module

func moduleParameters(m module, count int) []Tensor {
	params := make([]C.tensor, count)
	var err *C.char
	C.module_parameters(&err, C.module(m), (*C.tensor)(&params[0]))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := make([]Tensor, count)
	for i := 0; i < count; i++ {
		ret[i] = Tensor(params[i])
	}
	return ret
}

func ModuleToDevice(m module, device consts.DeviceType) {
	var err *C.char
	C.module_to_device(&err, C.module(m), C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
}

func ModuleToScalarType(m module, t consts.ScalarType) {
	var err *C.char
	C.module_to_scalar_type(&err, C.module(m), C.int8_t(t))
	if err != nil {
		panic(C.GoString(err))
	}
}

type Linear module

func NewLinear(inFeatures, outFeatures int64) Linear {
	var err *C.char
	l := C.new_linear(&err, C.int64_t(inFeatures), C.int64_t(outFeatures))
	if err != nil {
		panic(C.GoString(err))
	}
	return Linear(l)
}

func LinearForward(l Linear, x Tensor) Tensor {
	var err *C.char
	ptr := C.linear_forward(&err, C.module(l), C.tensor(x))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func LinearParameters(l Linear) []Tensor {
	return moduleParameters(module(l), 2)
}

type LayerNorm module

func NewLayerNorm(shape []int64) LayerNorm {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	l := C.new_layer_norm(&err, shapes, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return LayerNorm(l)
}

func LayerNormForward(l LayerNorm, x Tensor) Tensor {
	var err *C.char
	ptr := C.layer_norm_forward(&err, C.module(l), C.tensor(x))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func LayerNormParameters(l LayerNorm) []Tensor {
	return moduleParameters(module(l), 2)
}

type Attention module

func NewAttention(embedDim, numHeads int64, dropout float64) Attention {
	var err *C.char
	l := C.new_attention(&err, C.int64_t(embedDim), C.int64_t(numHeads), C.double(dropout))
	if err != nil {
		panic(C.GoString(err))
	}
	return Attention(l)
}

func AttentionForward(l Attention, q, k, v, mask Tensor, isCausal bool) (Tensor, Tensor) {
	if isCausal {
		mask = attentionBuildCausal(q, k)
		defer Free(mask)
	}
	var err *C.char
	var score C.tensor
	ptr := C.attention_forward(&err, C.module(l), C.tensor(q), C.tensor(k), C.tensor(v), C.tensor(mask), &score)
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr), Tensor(score)
}

func attentionBuildCausal(q, k Tensor) Tensor {
	l := q.Shapes()[0]
	s := k.Shapes()[0]
	mask := make([]float32, l*s)
	for i := int64(0); i < l; i++ {
		for j := int64(0); j < s; j++ {
			if j > i {
				mask[i*s+j] = float32(math.Inf(-1))
			}
		}
	}
	return FromFloat32(mask, []int64{l, s}, q.DeviceType())
}

func AttentionParameters(l Attention) []Tensor {
	return moduleParameters(module(l), 4)
}

type Embedding module

func NewEmbedding(numEmbeddings, embeddingDim, paddingIdx int64) Embedding {
	var err *C.char
	l := C.new_embedding(&err, C.int64_t(numEmbeddings), C.int64_t(embeddingDim), C.int64_t(paddingIdx))
	if err != nil {
		panic(C.GoString(err))
	}
	return Embedding(l)
}

func EmbeddingForward(l Embedding, x Tensor) Tensor {
	var err *C.char
	ptr := C.embedding_forward(&err, C.module(l), C.tensor(x))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func Parameters(l Embedding) []Tensor {
	return moduleParameters(module(l), 1)
}
