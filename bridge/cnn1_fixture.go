package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// CNN1Fixture holds 3D test inputs [samples][channels][seq_len].
type CNN1Fixture struct {
	XTest [][][]float64
}

func LoadCNN1Fixture(fixtureVersion, fixturesDir string) (*CNN1Fixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	flat, _, err := npzReadArray3D(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &CNN1Fixture{XTest: flat}, nil
}

func npzReadArray3D(npz []byte, name string) ([][][]float64, []int, error) {
	zr, err := openNPZ(npz)
	if err != nil {
		return nil, nil, err
	}
	raw, err := readNPZEntry(zr, name)
	if err != nil {
		return nil, nil, err
	}
	flat, shape, err := parseNPYFloat64(raw)
	if err != nil {
		return nil, nil, err
	}
	if len(shape) != 3 {
		return nil, shape, fmt.Errorf("npy %s: want rank-3 [N,C,L], got %v", name, shape)
	}
	n, c, l := shape[0], shape[1], shape[2]
	out := make([][][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([][]float64, c)
		for ch := 0; ch < c; ch++ {
			row := make([]float64, l)
			base := (i*c+ch)*l
			copy(row, flat[base:base+l])
			out[i][ch] = row
		}
	}
	return out, shape, nil
}

func SliceCNN1TestInputs(fx *CNN1Fixture, channels, seqLen int) [][][]float64 {
	out := make([][][]float64, len(fx.XTest))
	for i, sample := range fx.XTest {
		s := make([][]float64, channels)
		for ch := 0; ch < channels; ch++ {
			row := make([]float64, seqLen)
			if ch < len(sample) {
				copy(row, sample[ch][:seqLen])
			}
			s[ch] = row
		}
		out[i] = s
	}
	return out
}
