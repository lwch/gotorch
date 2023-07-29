package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
)

func TestScaledDotProductAttention(t *testing.T) {
	x := ARange(nil, 1*3*4, consts.KFloat).Reshape(1, 3, 4)
	y, score := ScaledDotProductAttention(x, x, x, nil, 0, true)
	fmt.Println(y.Float32Value())
	fmt.Println(score.Float32Value())
}

func TestPrint(t *testing.T) {
	x := ARange(nil, 1*3*4, consts.KFloat).Reshape(1, 3, 4)
	x.Print()
}