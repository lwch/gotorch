package nn

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/tensor"
)

func TestLinear(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	l := NewLinear(2, 3)
	x := tensor.ARange(s, 4, consts.KFloat).View(2, 2)
	y := l.Forward(x)
	fmt.Println(y.Float32Value())
	for _, p := range l.Parameters(s) {
		fmt.Println(p.Float32Value())
	}
}
