package nn

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/gotorch/mmgr"
	"github.com/lwch/gotorch/tensor"
)

func TestAttention(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	l := NewAttention(16, 4, 0.1)
	x := tensor.ARange(s, 2*2*16, consts.KFloat).View(2, 2, 16)
	y, score := l.Forward(x, x, x, nil, true)
	fmt.Println(y.Shapes(), y.Float32Value())
	fmt.Println(score.Shapes(), score.Float32Value())
	for _, p := range l.Parameters(s) {
		fmt.Println(p.Float32Value())
	}
}
