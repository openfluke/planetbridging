package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// EmbeddingFixture holds token-id test inputs [samples][seq_len].
type EmbeddingFixture struct {
	XTest [][]float64
}

func LoadEmbeddingFixture(fixtureVersion, fixturesDir string) (*EmbeddingFixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	flat, err := npzReadArray(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &EmbeddingFixture{XTest: flat}, nil
}

func SliceEmbeddingTestInputs(fx *EmbeddingFixture, seqLen int) [][]float64 {
	out := make([][]float64, len(fx.XTest))
	for i, sample := range fx.XTest {
		row := make([]float64, seqLen)
		copy(row, sample[:seqLen])
		out[i] = row
	}
	return out
}
