package tensor

import (
	"fmt"
	"testing"

	"github.com/lwch/gotorch/consts"
)

func TestBuildInfo(t *testing.T) {
	Zeros(consts.KFloat, WithShapes(2, 3))
	list := TensorInUse()
	for _, t := range list {
		fmt.Println(t.Created(), t.Trace())
	}
}
