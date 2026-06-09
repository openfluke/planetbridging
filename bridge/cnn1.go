package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// CNN1OutputLength computes conv1d output length (same as PyTorch).
func CNN1OutputLength(seqLen, kernel, stride, padding int) int {
	return (seqLen+2*padding-kernel)/stride + 1
}

// CNN1LayerStream is one Conv1d layer from a planet runtime.
type CNN1LayerStream struct {
	Kind        string    `json:"kind"`
	Index       int       `json:"index"`
	InChannels  int       `json:"in_channels"`
	Filters     int       `json:"filters"`
	InputLength int       `json:"input_length"`
	KernelSize  int       `json:"kernel_size"`
	Stride      int       `json:"stride"`
	Padding     int       `json:"padding"`
	Activation  string    `json:"activation"`
	Weights     []float64 `json:"weights"` // [filters × in_channels × kernel_size]
	Bias        []float64 `json:"bias,omitempty"`
}

// CNN1StreamRequest is POST body for CNN1 bedrock layer stream.
type CNN1StreamRequest struct {
	Bedrock        string            `json:"bedrock"`
	Planet         string            `json:"planet"`
	ModelID        string            `json:"model_id"`
	FixtureVersion string            `json:"fixture_version"`
	InputChannels  int               `json:"input_channels"`
	SeqLen         int               `json:"seq_len"`
	OutputDim      int               `json:"output_dim"`
	Layers         []CNN1LayerStream `json:"layers"`
}

func buildCNN1Layer(sl CNN1LayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "cnn1" {
		return fmt.Errorf("layer %d: want kind cnn1, got %q", layerIdx, sl.Kind)
	}
	outLen := CNN1OutputLength(sl.InputLength, sl.KernelSize, sl.Stride, sl.Padding)
	wantW := sl.Filters * sl.InChannels * sl.KernelSize
	if len(sl.Weights) != wantW {
		return fmt.Errorf("layer %d: weights len %d != %d×%d×%d",
			layerIdx, len(sl.Weights), sl.Filters, sl.InChannels, sl.KernelSize)
	}

	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerCNN1
	layer.Activation = manifestActivation(sl.Activation)
	layer.DType = poly.DTypeFloat32
	layer.InputChannels = sl.InChannels
	layer.InputHeight = sl.InputLength
	layer.Filters = sl.Filters
	layer.OutputHeight = outLen
	layer.KernelSize = sl.KernelSize
	layer.Stride = sl.Stride
	layer.Padding = sl.Padding
	layer.WeightStore = poly.NewWeightStore(wantW)
	for j, v := range sl.Weights {
		layer.WeightStore.Master[j] = float32(v)
	}
	return nil
}

// BuildNetworkFromCNN1Stream builds a volumetric CNN1 stack.
func BuildNetworkFromCNN1Stream(req CNN1StreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("cnn1 stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))

	inCh := req.InputChannels
	seqLen := req.SeqLen
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("cnn1 stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.InChannels != inCh {
			return nil, fmt.Errorf("cnn1 layer %d: in_channels %d != expected %d", i, sl.InChannels, inCh)
		}
		if sl.InputLength != seqLen {
			return nil, fmt.Errorf("cnn1 layer %d: input_length %d != expected %d", i, sl.InputLength, seqLen)
		}
		if err := buildCNN1Layer(sl, n, i); err != nil {
			return nil, err
		}
		seqLen = CNN1OutputLength(seqLen, sl.KernelSize, sl.Stride, sl.Padding)
		inCh = sl.Filters
	}
	return n, nil
}

// StreamCNN1ToEntity builds CNN1 network, saves .entity, infers on 3D fixture.
func StreamCNN1ToEntity(req CNN1StreamRequest, modelsDir string, fx *CNN1Fixture) (*StreamResult, error) {
	net, err := BuildNetworkFromCNN1Stream(req)
	if err != nil {
		return nil, err
	}

	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}

	lc, wb, _ := RoundTripEntity(entityPath)
	xTest := SliceCNN1TestInputs(fx, req.InputChannels, req.SeqLen)
	outs := InferCNN1Stack(net, xTest, req.OutputDim)

	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForCNN1Stream returns streamed CNN1 .entity path.
func EntityPathForCNN1Stream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
