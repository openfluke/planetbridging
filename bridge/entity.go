package bridge

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/openfluke/loom/poly"
)

// DenseBiases maps top-level layer index → per-output bias (stored outside Loom Dense blobs).
type DenseBiases map[int][]float32

type entityHeaderDoc struct {
	FormatVersion uint16                      `json:"format_version"`
	Network       poly.PersistenceNetworkSpec `json:"network"`
	Transformer   *poly.EntityTransformerSpec   `json:"transformer,omitempty"`
	Blobs         []poly.EntityWeightBlob     `json:"blobs"`
}

func SaveEntity(path string, net *poly.VolumetricNetwork, biases DenseBiases) error {
	data, err := poly.SerializeEntity(net)
	if err != nil {
		return err
	}
	if len(biases) > 0 {
		data, err = appendBiasBlobs(data, biases)
		if err != nil {
			return err
		}
	}
	return os.WriteFile(path, data, 0o644)
}

func LoadEntity(path string) (*poly.VolumetricNetwork, DenseBiases, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	net, err := poly.DeserializeEntity(data)
	if err != nil {
		return nil, nil, err
	}
	hdr, err := poly.ParseEntityHeader(data)
	if err != nil {
		return net, nil, err
	}
	return net, extractDenseBiases(hdr, data), nil
}

func WriteEntityFromNetwork(modelDir, planet, modelID, tag string, net *poly.VolumetricNetwork, biases DenseBiases) (string, error) {
	dir := filepath.Join(modelDir, planet, modelID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, modelID+"."+tag+".entity")
	if err := SaveEntity(path, net, biases); err != nil {
		return "", err
	}
	return path, nil
}

func RoundTripEntity(path string) (layerCount int, weightBytes int, err error) {
	net, _, err := LoadEntity(path)
	if err != nil {
		return 0, 0, err
	}
	for i := range net.Layers {
		if net.Layers[i].WeightStore != nil {
			weightBytes += len(net.Layers[i].WeightStore.Master) * 4
		}
	}
	return len(net.Layers), weightBytes, nil
}

func appendBiasBlobs(data []byte, biases DenseBiases) ([]byte, error) {
	hdr, err := poly.ParseEntityHeader(data)
	if err != nil {
		return nil, err
	}
	payload := append([]byte(nil), data[hdr.DataOffset:]...)
	doc := entityHeaderDoc{
		FormatVersion: hdr.FormatVersion,
		Network:       hdr.Network,
		Transformer:   hdr.Transformer,
		Blobs:         append([]poly.EntityWeightBlob(nil), hdr.Blobs...),
	}
	for idx, b := range biases {
		if len(b) == 0 {
			continue
		}
		raw := poly.EncodeWeightsRaw(b)
		offset := len(payload)
		payload = append(payload, raw...)
		doc.Blobs = append(doc.Blobs, poly.EntityWeightBlob{
			Path:   fmt.Sprintf("bridge.dense.%d.biases", idx),
			Offset: uint64(offset),
			Length: uint64(len(raw)),
			DType:  poly.DTypeFloat32.String(),
			Native: false,
		})
	}
	headerJSON, err := json.Marshal(doc)
	if err != nil {
		return nil, err
	}
	return assembleEntityFile(hdr.FormatVersion, hdr.Flags, headerJSON, payload), nil
}

func assembleEntityFile(version, flags uint16, headerJSON, payload []byte) []byte {
	out := make([]byte, 0, 20+len(headerJSON)+len(payload))
	out = append(out, []byte("ENTITY\x00\x00")...)
	var ver [2]byte
	binary.LittleEndian.PutUint16(ver[:], version)
	out = append(out, ver[:]...)
	var fl [2]byte
	binary.LittleEndian.PutUint16(fl[:], flags)
	out = append(out, fl[:]...)
	var hlen [8]byte
	binary.LittleEndian.PutUint64(hlen[:], uint64(len(headerJSON)))
	out = append(out, hlen[:]...)
	out = append(out, headerJSON...)
	out = append(out, payload...)
	return out
}

func extractDenseBiases(hdr *poly.EntityHeader, data []byte) DenseBiases {
	if hdr == nil {
		return nil
	}
	out := make(DenseBiases)
	for _, blob := range hdr.Blobs {
		if !strings.HasPrefix(blob.Path, "bridge.dense.") || !strings.HasSuffix(blob.Path, ".biases") {
			continue
		}
		parts := strings.Split(blob.Path, ".")
		if len(parts) < 4 {
			continue
		}
		idx, err := strconv.Atoi(parts[2])
		if err != nil {
			continue
		}
		end := int(blob.Offset) + int(blob.Length)
		if end > len(data)-hdr.DataOffset {
			continue
		}
		raw := data[hdr.DataOffset+int(blob.Offset) : hdr.DataOffset+end]
		b, err := poly.DecodeWeightsRaw(raw)
		if err != nil {
			continue
		}
		out[idx] = b
	}
	return out
}
