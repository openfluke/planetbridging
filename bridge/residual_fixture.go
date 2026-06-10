package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// ResidualFixture holds paired 3D test inputs [samples][seq_len][dim].
type ResidualFixture struct {
	MainTest [][][]float64
	SkipTest [][][]float64
}

func LoadResidualFixture(fixtureVersion, fixturesDir string) (*ResidualFixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	main, _, err := npzReadArray3D(b, "x_main_test")
	if err != nil {
		return nil, err
	}
	skip, _, err := npzReadArray3D(b, "x_skip_test")
	if err != nil {
		return nil, err
	}
	return &ResidualFixture{MainTest: main, SkipTest: skip}, nil
}

func SliceResidualTestInputs(x [][][]float64, seqLen, dim int) [][][]float64 {
	out := make([][][]float64, len(x))
	for i, sample := range x {
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
