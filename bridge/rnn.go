package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// RNNLayerStream is one vanilla RNN cell from a planet runtime.
type RNNLayerStream struct {
	Kind       string    `json:"kind"`
	Index      int       `json:"index"`
	InputSize  int       `json:"input_size"`
	HiddenSize int       `json:"hidden_size"`
	SeqLen     int       `json:"seq_len"`
	Weights    []float64 `json:"weights"` // [ih + hh + bias]
}

// RNNStreamRequest is POST body for RNN bedrock layer stream.
type RNNStreamRequest struct {
	Bedrock        string           `json:"bedrock"`
	Planet         string           `json:"planet"`
	ModelID        string           `json:"model_id"`
	FixtureVersion string           `json:"fixture_version"`
	InputSize      int              `json:"input_size"`
	HiddenSize     int              `json:"hidden_size"`
	SeqLen         int              `json:"seq_len"`
	OutputDim      int              `json:"output_dim"`
	Layers         []RNNLayerStream `json:"layers"`
}

func rnnWeightSize(inputSize, hiddenSize int) int {
	return hiddenSize*inputSize + hiddenSize*hiddenSize + hiddenSize
}

func packRNNWeights(sl RNNLayerStream) ([]float32, error) {
	inp, hid := sl.InputSize, sl.HiddenSize
	want := rnnWeightSize(inp, hid)
	if len(sl.Weights) != want {
		return nil, fmt.Errorf("weights len %d != %d", len(sl.Weights), want)
	}
	out := make([]float32, want)
	for i, v := range sl.Weights {
		out[i] = float32(v)
	}
	return out, nil
}

func buildRNNLayer(sl RNNLayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "rnn" {
		return fmt.Errorf("layer %d: want kind rnn, got %q", layerIdx, sl.Kind)
	}
	w, err := packRNNWeights(sl)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layerIdx, err)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerRNN
	layer.DType = poly.DTypeFloat32
	layer.InputHeight = sl.InputSize
	layer.OutputHeight = sl.HiddenSize
	layer.SeqLength = sl.SeqLen
	layer.WeightStore = poly.NewWeightStore(len(w))
	copy(layer.WeightStore.Master, w)
	return nil
}

// BuildNetworkFromRNNStream builds a volumetric RNN stack.
func BuildNetworkFromRNNStream(req RNNStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("rnn stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("rnn stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.InputSize != req.InputSize {
			return nil, fmt.Errorf("rnn layer %d: input_size %d != expected %d", i, sl.InputSize, req.InputSize)
		}
		if sl.HiddenSize != req.HiddenSize {
			return nil, fmt.Errorf("rnn layer %d: hidden_size %d != expected %d", i, sl.HiddenSize, req.HiddenSize)
		}
		if sl.SeqLen != req.SeqLen {
			return nil, fmt.Errorf("rnn layer %d: seq_len %d != expected %d", i, sl.SeqLen, req.SeqLen)
		}
		if err := buildRNNLayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamRNNToEntity builds RNN network, saves .entity, infers on fixture.
func StreamRNNToEntity(req RNNStreamRequest, modelsDir string, fx *RNNFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromRNNStream(req)
	if err != nil {
		return nil, err
	}
	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}
	lc, wb, _ := RoundTripEntity(entityPath)
	xTest := SliceRNNTestInputs(fx, req.SeqLen, req.InputSize)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = req.SeqLen * req.HiddenSize
	}
	outs := InferRNNStack(net, xTest, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForRNNStream returns streamed RNN .entity path.
func EntityPathForRNNStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
