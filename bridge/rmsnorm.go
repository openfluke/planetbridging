package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

const rmsNormDefaultEps = 1e-6

// RMSNormLayerStream is one RMSNorm layer from a planet runtime.
type RMSNormLayerStream struct {
	Kind    string    `json:"kind"`
	Index   int       `json:"index"`
	Dim     int       `json:"dim"`
	Weights []float64 `json:"weights"` // gamma [dim]
}

// RMSNormStreamRequest is POST body for RMSNorm bedrock layer stream.
type RMSNormStreamRequest struct {
	Bedrock        string               `json:"bedrock"`
	Planet         string               `json:"planet"`
	ModelID        string               `json:"model_id"`
	FixtureVersion string               `json:"fixture_version"`
	Dim            int                  `json:"dim"`
	SeqLen         int                  `json:"seq_len"`
	OutputDim      int                  `json:"output_dim"`
	Layers         []RMSNormLayerStream `json:"layers"`
}

func packRMSNormWeights(sl RMSNormLayerStream) ([]float32, error) {
	if len(sl.Weights) != sl.Dim {
		return nil, fmt.Errorf("weights len %d != %d", len(sl.Weights), sl.Dim)
	}
	out := make([]float32, sl.Dim)
	for i, v := range sl.Weights {
		out[i] = float32(v)
	}
	return out, nil
}

func buildRMSNormLayer(sl RMSNormLayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "rmsnorm" {
		return fmt.Errorf("layer %d: want kind rmsnorm, got %q", layerIdx, sl.Kind)
	}
	w, err := packRMSNormWeights(sl)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layerIdx, err)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerRMSNorm
	layer.Activation = poly.ActivationLinear
	layer.DType = poly.DTypeFloat32
	layer.InputHeight = sl.Dim
	layer.OutputHeight = sl.Dim
	layer.RMSNormEps = rmsNormDefaultEps
	layer.WeightStore = poly.NewWeightStore(len(w))
	copy(layer.WeightStore.Master, w)
	return nil
}

// BuildNetworkFromRMSNormStream builds a volumetric RMSNorm stack.
func BuildNetworkFromRMSNormStream(req RMSNormStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("rmsnorm stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("rmsnorm stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.Dim != req.Dim {
			return nil, fmt.Errorf("rmsnorm layer %d: dim %d != expected %d", i, sl.Dim, req.Dim)
		}
		if err := buildRMSNormLayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamRMSNormToEntity builds RMSNorm network, saves .entity, infers on fixture.
func StreamRMSNormToEntity(req RMSNormStreamRequest, modelsDir string, fx *RMSNormFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromRMSNormStream(req)
	if err != nil {
		return nil, err
	}
	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}
	lc, wb, _ := RoundTripEntity(entityPath)
	seqLen := req.SeqLen
	if seqLen == 0 {
		seqLen = 4
	}
	xTest := SliceRMSNormTestInputs(fx, seqLen, req.Dim)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = seqLen * req.Dim
	}
	outs := InferRMSNormStack(net, xTest, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForRMSNormStream returns streamed RMSNorm .entity path.
func EntityPathForRMSNormStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
