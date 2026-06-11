package bridge

import (
	"fmt"
	"os"
	"path/filepath"
)

// MixerFixture holds 5D CNN3-style test volumes and optional token ids for v2 embedding.
type MixerFixture struct {
	XTest     [][][][][]float64
	TokenTest [][]float64
}

func LoadMixerFixture(fixtureVersion, fixturesDir string) (*MixerFixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	cnn3, err := LoadCNN3Fixture(fixtureVersion, fixturesDir)
	if err != nil {
		return nil, err
	}
	tokens, _ := npzReadArray2D(b, "token_ids_test")
	return &MixerFixture{XTest: cnn3.XTest, TokenTest: tokens}, nil
}

func npzReadArray2D(npz []byte, name string) ([][]float64, error) {
	zr, err := openNPZ(npz)
	if err != nil {
		return nil, err
	}
	raw, err := readNPZEntry(zr, name)
	if err != nil {
		return nil, err
	}
	flat, shape, err := parseNPYFloat64(raw)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("npy %s: want rank-2, got %v", name, shape)
	}
	n, cols := shape[0], shape[1]
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, cols)
		base := i * cols
		copy(row, flat[base:base+cols])
		out[i] = row
	}
	return out, nil
}

func SliceMixerTestInputs(fx *MixerFixture) [][][][][]float64 {
	return SliceCNN3TestInputs(&CNN3Fixture{XTest: fx.XTest}, MixerVolumeC, MixerVolumeD, MixerVolumeH, MixerVolumeW)
}
