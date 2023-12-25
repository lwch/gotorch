package main

import (
	"fmt"
	"math/rand"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/loss"
	"github.com/lwch/gotorch/nn"
	"github.com/lwch/gotorch/optimizer"
	"github.com/lwch/gotorch/tensor"
)

const hiddenSize = 10
const device = consts.KCPU

var l1 = nn.NewLinear("l1", 2, hiddenSize)
var l2 = nn.NewLinear("l2", hiddenSize, 1)

func init() {
	l1.ToDevice(device)
	l2.ToDevice(device)
}

func main() {
	optm := optimizer.NewAdam()
	for i := 0; i < 1000; i++ {
		x, y := getBatch(true)
		// forward
		pred := forward(x)
		loss := loss.NewMse(pred, y)
		// backward
		loss.Backward()
		// update
		optm.Step(append(l1.Parameters(), l2.Parameters()...))
		if i%100 == 0 {
			fmt.Printf("epoch: %d loss: %f\n", i, loss.Value())
		}
	}
	x, _ := getBatch(false)
	pred := forward(x)
	fmt.Println("pred:", pred.ToDevice(consts.KCPU).Float32Value())
}

func forward(x *tensor.Tensor) *tensor.Tensor {
	y := l1.Forward(x)
	y = y.Relu()
	return l2.Forward(y)
}

func getBatch(shuffle bool) (*tensor.Tensor, *tensor.Tensor) {
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
	return tensor.FromFloat32("x", x,
			tensor.WithShapes(4, 2),
			tensor.WithDevice(device)),
		tensor.FromFloat32("y", y,
			tensor.WithShapes(4, 1),
			tensor.WithDevice(device))
}
