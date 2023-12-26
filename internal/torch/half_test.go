package torch

import (
	"testing"

	"github.com/lwch/gotorch/consts"
)

func TestHalf(t *testing.T) {
	ts := FromHalf([]float32{1, 2, 3, 4}, []int64{2, 2}, consts.KCPU)
	defer FreeTensor(ts)
	if ScalarType(ts) != consts.KHalf {
		t.Fatal("invalid scalar type")
	}
	if Dims(ts) != 2 {
		t.Fatal("invalid dim")
	}
	if Shapes(ts)[0] != 2 || Shapes(ts)[1] != 2 {
		t.Fatal("invalid shape")
	}
	if ElemCount(ts) != 4 {
		t.Fatal("invalid elem count")
	}
	values := HalfValue(ts)
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
	defer FreeTensor(ts)
	if ScalarType(ts) != consts.KBFloat16 {
		t.Fatal("invalid scalar type")
	}
	if Dims(ts) != 2 {
		t.Fatal("invalid dim")
	}
	if Shapes(ts)[0] != 2 || Shapes(ts)[1] != 2 {
		t.Fatal("invalid shape")
	}
	if ElemCount(ts) != 4 {
		t.Fatal("invalid elem count")
	}
	values := BFloat16Value(ts)
	if len(values) != 4 {
		t.Fatal("invalid value")
	}
	for i, v := range values {
		if v != float32(i+1) {
			t.Fatal("invalid value")
		}
	}
}
