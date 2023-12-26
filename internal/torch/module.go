package torch

// #include <stdint.h>
// #include "module.h"
import "C"
import (
	"math"
	"runtime"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/logging"
)

type NormalForward interface {
	Forward(Tensor) Tensor
}

type Module interface {
	Parameters() []Tensor
	ToDevice(consts.DeviceType)
	ToScalarType(consts.ScalarType)
}

type module struct {
	m C.module
}

func freeModule(m *module) error {
	if m.m == nil {
		return nil
	}
	logging.Debug("free module: %p", m)
	C.free_module(m.m)
	m.m = nil
	runtime.SetFinalizer(m, nil)
	return nil
}

func (m *module) Parameters(count int) []Tensor {
	params := make([]C.tensor, count)
	var err *C.char
	C.module_parameters(&err, m.m, (*C.tensor)(&params[0]))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := make([]Tensor, count)
	for i := 0; i < count; i++ {
		ret[i] = Tensor(params[i])
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
	*module
}

var _ Module = &Linear{}
var _ NormalForward = &Linear{}

func NewLinear(inFeatures, outFeatures int64) *Linear {
	var err *C.char
	l := C.new_linear(&err, C.int64_t(inFeatures), C.int64_t(outFeatures))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := &Linear{&module{l}}
	logging.Debug("new linear layer: %p", ret.module)
	runtime.SetFinalizer(ret.module, freeModule)
	return ret
}

func (l *Linear) Forward(x Tensor) Tensor {
	var err *C.char
	ptr := C.linear_forward(&err, l.m, C.tensor(x))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func (l *Linear) Parameters() []Tensor {
	return l.module.Parameters(2)
}

type LayerNorm struct {
	*module
}

var _ Module = &LayerNorm{}
var _ NormalForward = &LayerNorm{}

func NewLayerNorm(shape []int64) *LayerNorm {
	shapes, size := cInts[int64, C.int64_t](shape)
	var err *C.char
	l := C.new_layer_norm(&err, shapes, size)
	if err != nil {
		panic(C.GoString(err))
	}
	ret := &LayerNorm{&module{l}}
	logging.Debug("new layer_norm layer: %p", ret.module)
	runtime.SetFinalizer(ret.module, freeModule)
	return ret
}

func (l *LayerNorm) Forward(x Tensor) Tensor {
	var err *C.char
	ptr := C.layer_norm_forward(&err, l.m, C.tensor(x))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func (l *LayerNorm) Parameters() []Tensor {
	return l.module.Parameters(2)
}

type AttentionForward interface {
	Forward(Tensor, Tensor, Tensor, Tensor, bool) (Tensor, Tensor)
}

type Attention struct {
	*module
}

var _ Module = &Attention{}
var _ AttentionForward = &Attention{}

func NewAttention(embedDim, numHeads int64, dropout float64) *Attention {
	var err *C.char
	l := C.new_attention(&err, C.int64_t(embedDim), C.int64_t(numHeads), C.double(dropout))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := &Attention{&module{l}}
	logging.Debug("new attention layer: %p", ret.module)
	runtime.SetFinalizer(ret.module, freeModule)
	return ret
}

func (l *Attention) Forward(q, k, v, mask Tensor, isCausal bool) (Tensor, Tensor) {
	if isCausal {
		mask = attentionBuildCausal(q, k)
		defer FreeTensor(mask)
	}
	var err *C.char
	var score C.tensor
	ptr := C.attention_forward(&err, l.m, C.tensor(q), C.tensor(k), C.tensor(v), C.tensor(mask), &score)
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr), Tensor(score)
}

func attentionBuildCausal(q, k Tensor) Tensor {
	l := Shapes(q)[0]
	s := Shapes(k)[0]
	mask := make([]float32, l*s)
	for i := int64(0); i < l; i++ {
		for j := int64(0); j < s; j++ {
			if j > i {
				mask[i*s+j] = float32(math.Inf(-1))
			}
		}
	}
	return FromFloat32(mask, []int64{l, s}, DeviceType(q))
}

func (l *Attention) Parameters() []Tensor {
	return l.module.Parameters(4)
}

type Embedding struct {
	*module
}

var _ Module = &Embedding{}
var _ NormalForward = &Embedding{}

func NewEmbedding(numEmbeddings, embeddingDim, paddingIdx int64) *Embedding {
	var err *C.char
	l := C.new_embedding(&err, C.int64_t(numEmbeddings), C.int64_t(embeddingDim), C.int64_t(paddingIdx))
	if err != nil {
		panic(C.GoString(err))
	}
	ret := &Embedding{&module{l}}
	logging.Debug("new embedding layer: %p", ret.module)
	runtime.SetFinalizer(ret.module, freeModule)
	return ret
}

func (l *Embedding) Forward(x Tensor) Tensor {
	var err *C.char
	ptr := C.embedding_forward(&err, l.m, C.tensor(x))
	if err != nil {
		panic(C.GoString(err))
	}
	return Tensor(ptr)
}

func (l *Embedding) Parameters() []Tensor {
	return l.module.Parameters(1)
}
