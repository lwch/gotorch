package nn

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/tensor"
)

func TestLinear(t *testing.T) {
	l := NewLinear("linear", 2, 3)
	x := tensor.ARange("x", 4, consts.KFloat).View(2, 2)
	y := l.Forward(x)
	fmt.Println(y.Float32Value())
	for _, p := range l.Parameters() {
		fmt.Println(p.Float32Value())
	}
}

func TestLinearTo(t *testing.T) {
	l := NewLinear("linear", 2, 3)
	l.ToScalarType(consts.KDouble)
	for _, p := range l.Parameters() {
		fmt.Println(p.Float64Value())
	}
}
