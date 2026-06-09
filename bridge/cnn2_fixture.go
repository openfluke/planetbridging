package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// CNN2Fixture holds 4D test inputs [samples][channels][height][width].
type CNN2Fixture struct {
	XTest [][][][]float64
}

func LoadCNN2Fixture(fixtureVersion, fixturesDir string) (*CNN2Fixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	flat, _, err := npzReadArray4D(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &CNN2Fixture{XTest: flat}, nil
}

func npzReadArray4D(npz []byte, name string) ([][][][]float64, []int, error) {
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
	if len(shape) != 4 {
		return nil, shape, fmt.Errorf("npy %s: want rank-4 [N,C,H,W], got %v", name, shape)
	}
	n, c, h, w := shape[0], shape[1], shape[2], shape[3]
	out := make([][][][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([][][]float64, c)
		for ch := 0; ch < c; ch++ {
			plane := make([][]float64, h)
			for row := 0; row < h; row++ {
				line := make([]float64, w)
				base := ((i*c+ch)*h+row)*w
				copy(line, flat[base:base+w])
				plane[row] = line
			}
			out[i][ch] = plane
		}
	}
	return out, shape, nil
}

func SliceCNN2TestInputs(fx *CNN2Fixture, channels, height, width int) [][][][]float64 {
	out := make([][][][]float64, len(fx.XTest))
	for i, sample := range fx.XTest {
		s := make([][][]float64, channels)
		for ch := 0; ch < channels; ch++ {
			plane := make([][]float64, height)
			for row := 0; row < height; row++ {
				line := make([]float64, width)
				if ch < len(sample) && row < len(sample[ch]) {
					copy(line, sample[ch][row][:width])
				}
				plane[row] = line
			}
			s[ch] = plane
		}
		out[i] = s
	}
	return out
}
