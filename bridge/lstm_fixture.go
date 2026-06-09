package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// LSTMFixture holds 3D test inputs [samples][seq_len][input_size].
type LSTMFixture struct {
	XTest [][][]float64
}

func LoadLSTMFixture(fixtureVersion, fixturesDir string) (*LSTMFixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	flat, _, err := npzReadArray3D(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &LSTMFixture{XTest: flat}, nil
}

func SliceLSTMTestInputs(fx *LSTMFixture, seqLen, inputSize int) [][][]float64 {
	out := make([][][]float64, len(fx.XTest))
	for i, sample := range fx.XTest {
		s := make([][]float64, seqLen)
		for t := 0; t < seqLen; t++ {
			row := make([]float64, inputSize)
			if t < len(sample) {
				copy(row, sample[t][:inputSize])
			}
			s[t] = row
		}
		out[i] = s
	}
	return out
}
