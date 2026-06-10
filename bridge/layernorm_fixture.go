package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// LayerNormFixture holds 3D test inputs [samples][seq_len][dim].
type LayerNormFixture struct {
	XTest [][][]float64
}

func LoadLayerNormFixture(fixtureVersion, fixturesDir string) (*LayerNormFixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	flat, _, err := npzReadArray3D(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &LayerNormFixture{XTest: flat}, nil
}

func SliceLayerNormTestInputs(fx *LayerNormFixture, seqLen, dim int) [][][]float64 {
	out := make([][][]float64, len(fx.XTest))
	for i, sample := range fx.XTest {
		s := make([][]float64, seqLen)
		for t := 0; t < seqLen; t++ {
			row := make([]float64, dim)
			if t < len(sample) {
				copy(row, sample[t][:dim])
			}
			s[t] = row
		}
		out[i] = s
	}
	return out
}
