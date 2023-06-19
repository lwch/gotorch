package tensor

import (
	"math"
	"testing"

	"github.com/lwch/gotorch/mmgr"
)

func buildInts[T uint8 | int8 | int16 | int32 | int64]() []T {
	return []T{1, 2, 3, 4}
}

func compareInts[T uint8 | int8 | int16 | int32 | int64](a []T) bool {
	for i := T(1); i <= 4; i++ {
		if a[i-1] != i {
			return false
		}
	}
	return true
}

func buildFloats[T float32 | float64]() []T {
	return []T{1, 2, 3, 4}
}

func compareFloats[T float32 | float64](a []T) bool {
	for i := 1; i <= 4; i++ {
		if math.Abs(float64(a[i-1])-float64(i)) > 1e-3 {
			return false
		}
	}
	return true
}

func TestUint8(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromUint8(s, buildInts[uint8](), WithShapes(2, 2))
	v := ts.Uint8Value()
	if !compareInts(v) {
		t.Fail()
	}
}

func TestInt8(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromInt8(s, buildInts[int8](), WithShapes(2, 2))
	v := ts.Int8Value()
	if !compareInts(v) {
		t.Fail()
	}
}

func TestInt16(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromInt16(s, buildInts[int16](), WithShapes(2, 2))
	v := ts.Int16Value()
	if !compareInts(v) {
		t.Fail()
	}
}

func TestInt32(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromInt32(s, buildInts[int32](), WithShapes(2, 2))
	v := ts.Int32Value()
	if !compareInts(v) {
		t.Fail()
	}
}

func TestInt64(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromInt64(s, buildInts[int64](), WithShapes(2, 2))
	v := ts.Int64Value()
	if !compareInts(v) {
		t.Fail()
	}
}

func TestFloat32(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromFloat32(s, buildFloats[float32](), WithShapes(2, 2))
	v := ts.Float32Value()
	if !compareFloats(v) {
		t.Fail()
	}
}

func TestFloat64(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromFloat64(s, buildFloats[float64](), WithShapes(2, 2))
	v := ts.Float64Value()
	if !compareFloats(v) {
		t.Fail()
	}
}

func TestBool(t *testing.T) {
	s := mmgr.New()
	defer s.GC()
	ts := FromBool(s, []bool{true, true, false, false}, WithShapes(2, 2))
	v := ts.BoolValue()
	if v[0] != true ||
		v[1] != true ||
		v[2] != false ||
		v[3] != false {
		t.Fail()
	}
}
