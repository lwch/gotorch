package tensor

import (
	"runtime"
	"testing"

	"github.com/lwch/gotorch/consts"
	"github.com/lwch/logging"
)

func TestBuildInfo(t *testing.T) {
	func() {
		Zeros("zeros", consts.KFloat, WithShapes(2, 3))
		ShowLeaks()
		logging.Info("============================")
	}()
	runtime.GC() // tag
	runtime.GC() // gc
	ShowLeaks()
}
