package tensor

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

type build struct {
	created time.Time
	pcs     []uintptr
}

var leaks struct {
	sync.RWMutex
	m map[*Tensor]build
}

func init() {
	leaks.m = make(map[*Tensor]build)
}

func logBuild(t *Tensor) {
	pcs := make([]uintptr, 32)
	n := runtime.Callers(2, pcs)
	lk := build{
		created: time.Now(),
		pcs:     pcs[:n],
	}
	leaks.Lock()
	leaks.m[t] = lk
	leaks.Unlock()
}

func freeBuild(t *Tensor) {
	leaks.Lock()
	delete(leaks.m, t)
	leaks.Unlock()
}

type inUseTensor struct {
	t    *Tensor
	info build
}

func (t *inUseTensor) Tensor() *Tensor {
	return t.t
}

func (t *inUseTensor) Created() time.Time {
	return t.info.created
}

func (t *inUseTensor) Trace() []string {
	var ret []string
	frames := runtime.CallersFrames(t.info.pcs)
	for {
		frame, more := frames.Next()
		if !more {
			break
		}
		base := filepath.Base(frame.Function)
		if strings.HasPrefix(base, "tensor.") {
			continue
		}
		ret = append(ret, fmt.Sprintf("(%s:%d) %s", frame.File, frame.Line, frame.Function))
	}
	return ret
}

func TensorInUse() []*inUseTensor {
	leaks.RLock()
	defer leaks.RUnlock()
	ts := make([]*inUseTensor, len(leaks.m))
	i := 0
	for t, info := range leaks.m {
		ts[i] = &inUseTensor{
			t:    t,
			info: info,
		}
		i++
	}
	return ts
}
