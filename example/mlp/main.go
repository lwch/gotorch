package main

import (
	"fmt"
	"math/rand"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/loss"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
)

const hiddenSize = 10
const device = consts.KCPU

var w1, b1 *tensor.Tensor
var w2, b2 *tensor.Tensor

func randN(n int) []float32 {
	ret := make([]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = rand.Float32()
	}
	return ret
}

func init() {
	w1 = tensor.FromFloat32(nil, randN(2*hiddenSize),
		tensor.WithShapes(2, hiddenSize),
		tensor.WithDevice(device))
	b1 = tensor.Zeros(nil, consts.KFloat,
		tensor.WithShapes(hiddenSize),
		tensor.WithDevice(device))
	w2 = tensor.FromFloat32(nil, randN(hiddenSize*1),
		tensor.WithShapes(hiddenSize, 1),
		tensor.WithDevice(device))
	b2 = tensor.Zeros(nil, consts.KFloat,
		tensor.WithShapes(1),
		tensor.WithDevice(device))
	w1.SetRequiresGrad(true)
	b1.SetRequiresGrad(true)
	w2.SetRequiresGrad(true)
	b2.SetRequiresGrad(true)
}

func main() {
	s := mmgr.New()
	defer s.GC()
	optm := optimizer.NewAdam()
	for i := 0; i < 10000; i++ {
		x, y := getBatch(s, true)
		// forward
		pred := forward(x)
		loss := loss.NewMse(pred, y)
		// backward
		loss.Backward()
		// update
		optm.Step([]*tensor.Tensor{w1, b1, w2, b2})
		if i%100 == 0 {
			fmt.Printf("epoch: %d loss: %f\n", i, loss.Value())
			s.GC()
		}
	}
	x, _ := getBatch(s, false)
	pred := forward(x)
	fmt.Println("pred:", pred.ToDevice(consts.KCPU).Float32Value())
}

func forward(x *tensor.Tensor) *tensor.Tensor {
	y := x.MatMul(w1).Add(b1)
	y = y.Relu()
	return y.MatMul(w2).Add(b2)
}

func getBatch(s *mmgr.Storage, shuffle bool) (*tensor.Tensor, *tensor.Tensor) {
	x := []float32{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}
	y := []float32{
		0,
		1,
		1,
		0,
	}
	if shuffle {
		rand.Shuffle(4, func(i, j int) {
			x[i*2], x[j*2] = x[j*2], x[i*2]
			x[i*2+1], x[j*2+1] = x[j*2+1], x[i*2+1]
			y[i], y[j] = y[j], y[i]
		})
	}
	return tensor.FromFloat32(s, x,
			tensor.WithShapes(4, 2),
			tensor.WithDevice(device)),
		tensor.FromFloat32(s, y,
			tensor.WithShapes(4, 1),
			tensor.WithDevice(device))
}
