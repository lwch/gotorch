package torch

import (
	"testing"

	"github.com/lwch/gotorch/consts"
)

func TestHalf(t *testing.T) {
	ts := FromHalf([]float32{1, 2, 3, 4}, []int64{2, 2}, consts.KCPU)
	defer ts.Free()
	if ts.ScalarType() != consts.KHalf {
		t.Fatal("invalid scalar type")
	}
	if ts.Dims() != 2 {
		t.Fatal("invalid dim")
	}
	if ts.Shapes()[0] != 2 || ts.Shapes()[1] != 2 {
		t.Fatal("invalid shape")
	}
	if ts.ElemCount() != 4 {
		t.Fatal("invalid elem count")
	}
	values := ts.HalfValue()
	if len(values) != 4 {
		t.Fatal("invalid value")
	}
	for i, v := range values {
		if v != float32(i+1) {
			t.Fatal("invalid value")
		}
	}
}

func TestBFloat16(t *testing.T) {
	ts := FromBFloat16([]float32{1, 2, 3, 4}, []int64{2, 2}, consts.KCPU)
	defer ts.Free()
	if ts.ScalarType() != consts.KBFloat16 {
		t.Fatal("invalid scalar type")
	}
	if ts.Dims() != 2 {
		t.Fatal("invalid dim")
	}
	if ts.Shapes()[0] != 2 || ts.Shapes()[1] != 2 {
		t.Fatal("invalid shape")
	}
	if ts.ElemCount() != 4 {
		t.Fatal("invalid elem count")
	}
	values := ts.BFloat16Value()
	if len(values) != 4 {
		t.Fatal("invalid value")
	}
	for i, v := range values {
		if v != float32(i+1) {
			t.Fatal("invalid value")
		}
	}
}
