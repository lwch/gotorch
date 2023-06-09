package mmgr

import (
	"sync"

	"github.com/lwch/gotorch/internal/torch"
)

type Storage struct {
	m    sync.Mutex
	data []*torch.Tensor
}

func New() *Storage {
	return &Storage{}
}

func (s *Storage) Put(t *torch.Tensor) {
	s.m.Lock()
	defer s.m.Unlock()
	s.data = append(s.data, t)
}

func (s *Storage) GC() {
	s.m.Lock()
	defer s.m.Unlock()
	for _, t := range s.data {
		t.Free()
	}
	s.data = s.data[:0]
}
