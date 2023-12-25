package tensor

import (
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
)

var leaks struct {
	sync.RWMutex
	data map[uint64][]uintptr // don't ref to *Tensor
	idx  atomic.Uint64
}

func init() {
	leaks.data = make(map[uint64][]uintptr)
}

func logBuildInfo(t *Tensor) {
	idx := leaks.idx.Add(1)
	t.idx = idx
	pcs := make([]uintptr, 32)
	n := runtime.Callers(2, pcs)
	leaks.Lock()
	leaks.data[idx] = pcs[:n]
	leaks.Unlock()
}

func free(t *Tensor) {
	leaks.Lock()
	delete(leaks.data, t.idx)
	leaks.Unlock()
}

func WriteLeaks(w io.Writer) {
	raw := make(map[uint64][]uintptr, len(leaks.data))
	leaks.RLock()
	for idx, pcs := range leaks.data {
		raw[idx] = pcs
	}
	leaks.RUnlock()
	writeStack := func(pcs []uintptr) {
		frames := runtime.CallersFrames(pcs)
		for {
			frame, more := frames.Next()
			base := filepath.Base(frame.Function)
			if strings.HasPrefix(base, "tensor.") {
				continue
			}
			fmt.Fprintf(w, "  - %s:%d => %s\n", frame.File, frame.Line, frame.Function)
			if !more {
				break
			}
		}
	}
	for idx, pcs := range raw {
		fmt.Fprintf(w, "tensor [>> tensor.%d <<] leaked:\n", idx)
		writeStack(pcs)
	}
}
