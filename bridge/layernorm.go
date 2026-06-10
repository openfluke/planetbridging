package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// LayerNormLayerStream is one LayerNorm layer from a planet runtime.
type LayerNormLayerStream struct {
	Kind    string    `json:"kind"`
	Index   int       `json:"index"`
	Dim     int       `json:"dim"`
	Weights []float64 `json:"weights"` // [gamma | beta] (2×dim)
}

// LayerNormStreamRequest is POST body for LayerNorm bedrock layer stream.
type LayerNormStreamRequest struct {
	Bedrock        string                 `json:"bedrock"`
	Planet         string                 `json:"planet"`
	ModelID        string                 `json:"model_id"`
	FixtureVersion string                 `json:"fixture_version"`
	Dim            int                    `json:"dim"`
	SeqLen         int                    `json:"seq_len"`
	OutputDim      int                    `json:"output_dim"`
	Layers         []LayerNormLayerStream `json:"layers"`
}

func layerNormWeightSize(dim int) int {
	return 2 * dim
}

func packLayerNormWeights(sl LayerNormLayerStream) ([]float32, error) {
	want := layerNormWeightSize(sl.Dim)
	if len(sl.Weights) != want {
		return nil, fmt.Errorf("weights len %d != %d", len(sl.Weights), want)
	}
	out := make([]float32, want)
	for i, v := range sl.Weights {
		out[i] = float32(v)
	}
	return out, nil
}

func buildLayerNormLayer(sl LayerNormLayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "layernorm" {
		return fmt.Errorf("layer %d: want kind layernorm, got %q", layerIdx, sl.Kind)
	}
	w, err := packLayerNormWeights(sl)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layerIdx, err)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerLayerNorm
	layer.Activation = poly.ActivationLinear
	layer.DType = poly.DTypeFloat32
	layer.InputHeight = sl.Dim
	layer.OutputHeight = sl.Dim
	layer.WeightStore = poly.NewWeightStore(len(w))
	copy(layer.WeightStore.Master, w)
	return nil
}

// BuildNetworkFromLayerNormStream builds a volumetric LayerNorm stack.
func BuildNetworkFromLayerNormStream(req LayerNormStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("layernorm stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("layernorm stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.Dim != req.Dim {
			return nil, fmt.Errorf("layernorm layer %d: dim %d != expected %d", i, sl.Dim, req.Dim)
		}
		if err := buildLayerNormLayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamLayerNormToEntity builds LayerNorm network, saves .entity, infers on fixture.
func StreamLayerNormToEntity(req LayerNormStreamRequest, modelsDir string, fx *LayerNormFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromLayerNormStream(req)
	if err != nil {
		return nil, err
	}
	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}
	lc, wb, _ := RoundTripEntity(entityPath)
	seqLen := req.SeqLen
	if seqLen == 0 && len(req.Layers) > 0 {
		seqLen = 4
	}
	xTest := SliceLayerNormTestInputs(fx, seqLen, req.Dim)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = seqLen * req.Dim
	}
	outs := InferLayerNormStack(net, xTest, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForLayerNormStream returns streamed LayerNorm .entity path.
func EntityPathForLayerNormStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
