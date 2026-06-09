package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// LSTMLayerStream is one LSTM cell from a planet runtime.
type LSTMLayerStream struct {
	Kind       string    `json:"kind"`
	Index      int       `json:"index"`
	InputSize  int       `json:"input_size"`
	HiddenSize int       `json:"hidden_size"`
	SeqLen     int       `json:"seq_len"`
	IWeights   []float64 `json:"i_weights"` // [ih + hh + bias] per gate
	FWeights   []float64 `json:"f_weights"`
	GWeights   []float64 `json:"g_weights"`
	OWeights   []float64 `json:"o_weights"`
}

// LSTMStreamRequest is POST body for LSTM bedrock layer stream.
type LSTMStreamRequest struct {
	Bedrock        string            `json:"bedrock"`
	Planet         string            `json:"planet"`
	ModelID        string            `json:"model_id"`
	FixtureVersion string            `json:"fixture_version"`
	InputSize      int               `json:"input_size"`
	HiddenSize     int               `json:"hidden_size"`
	SeqLen         int               `json:"seq_len"`
	OutputDim      int               `json:"output_dim"`
	Layers         []LSTMLayerStream `json:"layers"`
}

func lstmGateSize(inputSize, hiddenSize int) int {
	return hiddenSize*inputSize + hiddenSize*hiddenSize + hiddenSize
}

func packLSTMWeights(sl LSTMLayerStream) ([]float32, error) {
	inp, hid := sl.InputSize, sl.HiddenSize
	gate := lstmGateSize(inp, hid)
	for _, w := range []struct {
		label string
		data  []float64
	}{
		{"i", sl.IWeights},
		{"f", sl.FWeights},
		{"g", sl.GWeights},
		{"o", sl.OWeights},
	} {
		if len(w.data) != gate {
			return nil, fmt.Errorf("%s_weights len %d != %d", w.label, len(w.data), gate)
		}
	}
	total := 4 * gate
	out := make([]float32, total)
	n := 0
	for _, w := range [][]float64{sl.IWeights, sl.FWeights, sl.GWeights, sl.OWeights} {
		for _, v := range w {
			out[n] = float32(v)
			n++
		}
	}
	return out, nil
}

func buildLSTMLayer(sl LSTMLayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "lstm" {
		return fmt.Errorf("layer %d: want kind lstm, got %q", layerIdx, sl.Kind)
	}
	w, err := packLSTMWeights(sl)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layerIdx, err)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerLSTM
	layer.DType = poly.DTypeFloat32
	layer.InputHeight = sl.InputSize
	layer.OutputHeight = sl.HiddenSize
	layer.SeqLength = sl.SeqLen
	layer.WeightStore = poly.NewWeightStore(len(w))
	copy(layer.WeightStore.Master, w)
	return nil
}

// BuildNetworkFromLSTMStream builds a volumetric LSTM stack.
func BuildNetworkFromLSTMStream(req LSTMStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("lstm stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("lstm stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.InputSize != req.InputSize {
			return nil, fmt.Errorf("lstm layer %d: input_size %d != expected %d", i, sl.InputSize, req.InputSize)
		}
		if sl.HiddenSize != req.HiddenSize {
			return nil, fmt.Errorf("lstm layer %d: hidden_size %d != expected %d", i, sl.HiddenSize, req.HiddenSize)
		}
		if sl.SeqLen != req.SeqLen {
			return nil, fmt.Errorf("lstm layer %d: seq_len %d != expected %d", i, sl.SeqLen, req.SeqLen)
		}
		if err := buildLSTMLayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamLSTMToEntity builds LSTM network, saves .entity, infers on fixture.
func StreamLSTMToEntity(req LSTMStreamRequest, modelsDir string, fx *LSTMFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromLSTMStream(req)
	if err != nil {
		return nil, err
	}
	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}
	lc, wb, _ := RoundTripEntity(entityPath)
	xTest := SliceLSTMTestInputs(fx, req.SeqLen, req.InputSize)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = req.SeqLen * req.HiddenSize
	}
	outs := InferLSTMStack(net, xTest, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForLSTMStream returns streamed LSTM .entity path.
func EntityPathForLSTMStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
