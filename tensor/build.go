package tensor

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/lwch/logging"
)

var leaks struct {
	sync.RWMutex
	data map[string][]uintptr // don't ref to *Tensor
	idx  atomic.Uint64
}

func init() {
	leaks.data = make(map[string][]uintptr)
}

func logBuildInfo(t *Tensor) {
	if len(t.name) != 0 {
		panic("tensor name must be empty")
	}
	t.name = fmt.Sprintf("ts.%d", leaks.idx.Add(1))
	pcs := make([]uintptr, 32)
	n := runtime.Callers(2, pcs)
	leaks.Lock()
	leaks.data[t.name] = pcs[:n]
	leaks.Unlock()
}

func free(t *Tensor) {
	leaks.Lock()
	delete(leaks.data, t.name)
	leaks.Unlock()
}

func ShowLeaks() {
	raw := make(map[string][]uintptr, len(leaks.data))
	leaks.RLock()
	for name, pcs := range leaks.data {
		raw[name] = pcs
	}
	leaks.RUnlock()
	stack := func(pcs []uintptr) string {
		var rows []string
		frames := runtime.CallersFrames(pcs)
		for {
			frame, more := frames.Next()
			base := filepath.Base(frame.Function)
			if strings.HasPrefix(base, "tensor.") {
				continue
			}
			rows = append(rows, fmt.Sprintf("  - %s:%d => %s", frame.File, frame.Line, frame.Function))
			if !more {
				break
			}
		}
		return strings.Join(rows, "\n")
	}
	for name, pcs := range raw {
		logging.Info("tensor [>> %s <<] leaked:\n%s", name, stack(pcs))
	}
}
