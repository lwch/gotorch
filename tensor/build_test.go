package tensor

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/lwch/gotorch/consts"
)

func TestBuildInfo(t *testing.T) {
	var buf bytes.Buffer
	func() {
		Zeros(consts.KFloat, WithShapes(2, 3))
		WriteLeaks(&buf)
		if buf.Len() == 0 {
			t.Fatal("no memory leak")
		}
	}()
	runtime.GC() // tag
	runtime.GC() // gc
	buf.Reset()
	WriteLeaks(&buf)
	if buf.Len() != 0 {
		t.Fatal("memory leak")
	}
}
