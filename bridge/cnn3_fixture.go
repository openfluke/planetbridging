package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// CNN3Fixture holds 5D test inputs [samples][channels][depth][height][width].
type CNN3Fixture struct {
	XTest [][][][][]float64
}

func LoadCNN3Fixture(fixtureVersion, fixturesDir string) (*CNN3Fixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	flat, _, err := npzReadArray5D(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &CNN3Fixture{XTest: flat}, nil
}

func npzReadArray5D(npz []byte, name string) ([][][][][]float64, []int, error) {
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
	if len(shape) != 5 {
		return nil, shape, fmt.Errorf("npy %s: want rank-5 [N,C,D,H,W], got %v", name, shape)
	}
	n, c, d, h, w := shape[0], shape[1], shape[2], shape[3], shape[4]
	out := make([][][][][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([][][][]float64, c)
		for ch := 0; ch < c; ch++ {
			vol := make([][][]float64, d)
			for dep := 0; dep < d; dep++ {
				plane := make([][]float64, h)
				for row := 0; row < h; row++ {
					line := make([]float64, w)
					base := (((i*c+ch)*d+dep)*h+row)*w
					copy(line, flat[base:base+w])
					plane[row] = line
				}
				vol[dep] = plane
			}
			out[i][ch] = vol
		}
	}
	return out, shape, nil
}

func SliceCNN3TestInputs(fx *CNN3Fixture, channels, depth, height, width int) [][][][][]float64 {
	out := make([][][][][]float64, len(fx.XTest))
	for i, sample := range fx.XTest {
		s := make([][][][]float64, channels)
		for ch := 0; ch < channels; ch++ {
			vol := make([][][]float64, depth)
			for dep := 0; dep < depth; dep++ {
				plane := make([][]float64, height)
				for row := 0; row < height; row++ {
					line := make([]float64, width)
					if ch < len(sample) && dep < len(sample[ch]) && row < len(sample[ch][dep]) {
						copy(line, sample[ch][dep][row][:width])
					}
					plane[row] = line
				}
				vol[dep] = plane
			}
			s[ch] = vol
		}
		out[i] = s
	}
	return out
}
