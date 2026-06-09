package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// MHAFixture holds 3D test inputs [samples][seq_len][d_model].
type MHAFixture struct {
	XTest [][][]float64
}

func LoadMHAFixture(fixtureVersion, fixturesDir string) (*MHAFixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	flat, _, err := npzReadArray3D(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &MHAFixture{XTest: flat}, nil
}

func SliceMHATestInputs(fx *MHAFixture, seqLen, dModel int) [][][]float64 {
	out := make([][][]float64, len(fx.XTest))
	for i, sample := range fx.XTest {
		s := make([][]float64, seqLen)
		for t := 0; t < seqLen; t++ {
			row := make([]float64, dModel)
			if t < len(sample) {
				copy(row, sample[t][:dModel])
			}
			s[t] = row
		}
		out[i] = s
	}
	return out
}
