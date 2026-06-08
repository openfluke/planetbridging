package bridge

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// Fixture holds shared bedrock test inputs (from generated npz).
type Fixture struct {
	XTest [][]float64
}

func LoadFixture(fixtureVersion, fixturesDir string) (*Fixture, error) {
	path := filepath.Join(fixturesDir, fixtureVersion+".npz")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("fixture %s: %w", path, err)
	}
	xTest, err := npzReadArray(b, "x_test")
	if err != nil {
		return nil, err
	}
	return &Fixture{XTest: xTest}, nil
}

func SliceTestInputs(fx *Fixture, inputDim int) [][]float64 {
	out := make([][]float64, len(fx.XTest))
	for i, row := range fx.XTest {
		s := make([]float64, inputDim)
		copy(s, row[:inputDim])
		out[i] = s
	}
	return out
}

func npzReadArray(npz []byte, name string) ([][]float64, error) {
	zr, err := zip.NewReader(bytes.NewReader(npz), int64(len(npz)))
	if err != nil {
		return nil, err
	}
	var raw []byte
	for _, f := range zr.File {
		if f.Name != name+".npy" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		buf, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		raw = buf
		break
	}
	if raw == nil {
		return nil, fmt.Errorf("npz: missing %s.npy", name)
	}
	flat, shape, err := parseNPYFloat64(raw)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("npy %s: want rank-2, got %v", name, shape)
	}
	rows, cols := shape[0], shape[1]
	out := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		out[r] = flat[r*cols : (r+1)*cols]
	}
	return out, nil
}

func parseNPYFloat64(raw []byte) ([]float64, []int, error) {
	if len(raw) < 10 || string(raw[:6]) != "\x93NUMPY" {
		return nil, nil, fmt.Errorf("invalid npy header")
	}
	major := raw[6]
	var headerLen int
	var off int
	if major == 1 {
		headerLen = int(binary.LittleEndian.Uint16(raw[8:10]))
		off = 10
	} else {
		headerLen = int(binary.LittleEndian.Uint32(raw[8:12]))
		off = 12
	}
	header := string(raw[off : off+headerLen])
	shape, err := npyParseShape(header)
	if err != nil {
		return nil, nil, err
	}
	dataOff := off + headerLen
	data := raw[dataOff:]
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(data) < n*8 {
		return nil, nil, fmt.Errorf("npy data truncated")
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return out, shape, nil
}

func npyParseShape(header string) ([]int, error) {
	i := stringsIndex(header, "'shape':")
	if i < 0 {
		return nil, fmt.Errorf("npy header missing shape")
	}
	rest := header[i+8:]
	start := stringsIndex(rest, "(")
	end := stringsIndex(rest, ")")
	if start < 0 || end < 0 || end <= start {
		return nil, fmt.Errorf("npy header bad shape")
	}
	inner := rest[start+1 : end]
	var shape []int
	for _, p := range splitComma(inner) {
		p = trimSpace(p)
		if p == "" {
			continue
		}
		var v int
		fmt.Sscanf(p, "%d", &v)
		shape = append(shape, v)
	}
	return shape, nil
}

func stringsIndex(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func splitComma(s string) []string {
	var out []string
	cur := ""
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			out = append(out, cur)
			cur = ""
			continue
		}
		cur += string(s[i])
	}
	out = append(out, cur)
	return out
}

func trimSpace(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\n') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\n') {
		s = s[:len(s)-1]
	}
	return s
}
