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

var sUnfree = mmgr.New()

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
	w1 = tensor.FromFloat32(sUnfree, randN(2*hiddenSize), 2, hiddenSize)
	b1 = tensor.Zeros(sUnfree, consts.KFloat, 10)
	w2 = tensor.FromFloat32(sUnfree, randN(hiddenSize*1), hiddenSize, 1)
	b2 = tensor.Zeros(sUnfree, consts.KFloat, 1)
	w1.SetRequiresGrad(true)
	b1.SetRequiresGrad(true)
	w2.SetRequiresGrad(true)
	b2.SetRequiresGrad(true)
}

func main() {
	s := mmgr.New()
	defer s.GC()
	optm := optimizer.NewAdam()
	for i := 0; i < 5000; i++ {
		x, y := getBatch(s)
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
	fmt.Println("w1=", w1.Float32Value())
	fmt.Println("b1=", b1.Float32Value())
	fmt.Println("w2=", w2.Float32Value())
	fmt.Println("b2=", b2.Float32Value())
	x, _ := getBatch(s)
	pred := forward(x)
	fmt.Println(pred.Float32Value())
}

func forward(x *tensor.Tensor) *tensor.Tensor {
	y := x.MatMul(w1).Add(b1)
	y = y.Relu()
	return y.MatMul(w2).Add(b2)
}

func getBatch(s *mmgr.Storage) (*tensor.Tensor, *tensor.Tensor) {
	return tensor.FromFloat32(s, []float32{
			0, 0,
			0, 1,
			1, 0,
			1, 1,
		}, 4, 2), tensor.FromFloat32(s, []float32{
			0,
			1,
			1,
			0,
		}, 4, 1)
}
