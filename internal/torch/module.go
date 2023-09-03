package torch

// #include <stdint.h>
// #include "module.h"
import "C"
import (
	"math"

	"github.com/lwch/gotorch/consts"
)

type NormalForward interface {
	Forward(*Tensor) *Tensor
}

type Module interface {
	Parameters() []*Tensor
	ToDevice(consts.DeviceType)
	ToScalarType(consts.ScalarType)
}

type module struct {
	m C.module
}

func (m *module) parameters(count int) []*Tensor {
	params := make([]C.tensor, count)
	var err *C.char
	C.module_parameters(&err, m.m, (*C.tensor)(&params[0]))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := make([]*Tensor, count)
	for i := 0; i < count; i++ {
		ret[i] = &Tensor{data: params[i]}
	}
	return ret
}

func (m *module) ToDevice(device consts.DeviceType) {
	var err *C.char
	C.module_to_device(&err, m.m, C.int8_t(device))
	if err != nil {
		panic(C.GoString(err))
	}
}

func (m *module) ToScalarType(t consts.ScalarType) {
	var err *C.char
	C.module_to_scalar_type(&err, m.m, C.int8_t(t))
	if err != nil {
		panic(C.GoString(err))
	}
}

type Linear struct {
	module
}

func NewLinear(inFeatures, outFeatures int64) *Linear {
	var err *C.char
	l := C.new_linear(&err, C.int64_t(inFeatures), C.int64_t(outFeatures))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Linear{module{l}}
}

func (l *Linear) Forward(x *Tensor) *Tensor {
	var err *C.char
	ptr := C.linear_forward(&err, l.m, x.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (l *Linear) Parameters() []*Tensor {
	return l.parameters(2)
}

type LayerNorm struct {
	module
}

func NewLayerNorm(shape []int64) *LayerNorm {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	l := C.new_layer_norm(&err, shapes, size)
	if err != nil {
		panic(C.GoString(err))
	}
	return &LayerNorm{module{l}}
}

func (l *LayerNorm) Forward(x *Tensor) *Tensor {
	var err *C.char
	ptr := C.layer_norm_forward(&err, l.m, x.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (l *LayerNorm) Parameters() []*Tensor {
	return l.parameters(2)
}

type AttentionForward interface {
	Forward(*Tensor, *Tensor, *Tensor, *Tensor, bool) (*Tensor, *Tensor)
}

type Attention struct {
	module
}

func NewAttention(embedDim, numHeads int64, dropout float64) *Attention {
	var err *C.char
	l := C.new_attention(&err, C.int64_t(embedDim), C.int64_t(numHeads), C.double(dropout))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Attention{module{l}}
}

func (l *Attention) Forward(q, k, v, mask *Tensor, isCausal bool) (*Tensor, *Tensor) {
	if isCausal {
		mask = l.buildCausal(q, k)
		defer mask.Free()
	}
	var err *C.char
	var score C.tensor
	var m C.tensor
	if mask != nil {
		m = mask.data
	}
	ptr := C.attention_forward(&err, l.m, q.data, k.data, v.data, m, &score)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}, &Tensor{data: score}
}

func (*Attention) buildCausal(q, k *Tensor) *Tensor {
	l := q.Shapes()[q.Dims()-2]
	s := k.Shapes()[k.Dims()-2]
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

func (l *Attention) Parameters() []*Tensor {
	return l.parameters(4)
}

type Embedding struct {
	module
}

func NewEmbedding(numEmbeddings, embeddingDim, paddingIdx int64) *Embedding {
	var err *C.char
	l := C.new_embedding(&err, C.int64_t(numEmbeddings), C.int64_t(embeddingDim), C.int64_t(paddingIdx))
	if err != nil {
		panic(C.GoString(err))
	}
	return &Embedding{module{l}}
}

func (l *Embedding) Forward(x *Tensor) *Tensor {
	var err *C.char
	ptr := C.embedding_forward(&err, l.m, x.data)
	if err != nil {
		panic(C.GoString(err))
	}
	return &Tensor{data: ptr}
}

func (l *Embedding) Parameters() []*Tensor {
	return l.parameters(1)
}
