package tensor

import (
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"sort"
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
	if _, ok := leaks.data[t.idx]; !ok {
		panic("tensor: double free")
	}
	delete(leaks.data, t.idx)
	leaks.Unlock()
}

func WriteLeaks(w io.Writer) {
	raw := make(map[uint64][]uintptr, len(leaks.data))
	ids := make([]uint64, 0, len(leaks.data))
	leaks.RLock()
	for idx, pcs := range leaks.data {
		raw[idx] = pcs
		ids = append(ids, idx)
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
			if strings.HasPrefix(base, "runtime.") {
				continue
			}
			fmt.Fprintf(w, "  - %s:%d => %s\n", frame.File, frame.Line, frame.Function)
			if !more {
				break
			}
		}
	}
	sort.Slice(ids, func(i, j int) bool {
		return ids[i] < ids[j]
	})
	for _, id := range ids {
		fmt.Fprintf(w, "tensor [>> tensor.%d <<] leaked:\n", id)
		writeStack(raw[id])
	}
}
