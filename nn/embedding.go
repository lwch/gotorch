package nn

import (
	"github.com/lwch/gotorch/internal/torch"
)

type Embedding struct {
	module
}

func NewEmbedding(name string, numEmbeddings, embeddingDim, paddingIdx int64) *Embedding {
	return &Embedding{
		module{
			name: name,
			m:    torch.NewEmbedding(numEmbeddings, embeddingDim, paddingIdx),
		},
	}
}
