package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// SwiGLULayerStream is one SwiGLU layer from a planet runtime.
type SwiGLULayerStream struct {
	Kind             string    `json:"kind"`
	Index            int       `json:"index"`
	InputDim         int       `json:"input_dim"`
	IntermediateDim  int       `json:"intermediate_dim"`
	Weights          []float64 `json:"weights"` // gateW|upW|downW|gateB|upB|downB
}

// SwiGLUStreamRequest is POST body for SwiGLU bedrock layer stream.
type SwiGLUStreamRequest struct {
	Bedrock        string              `json:"bedrock"`
	Planet         string              `json:"planet"`
	ModelID        string              `json:"model_id"`
	FixtureVersion string              `json:"fixture_version"`
	InputDim       int                 `json:"input_dim"`
	IntermediateDim int                `json:"intermediate_dim"`
	SeqLen         int                 `json:"seq_len"`
	OutputDim      int                 `json:"output_dim"`
	Layers         []SwiGLULayerStream `json:"layers"`
}

func swigluWeightSize(inputDim, interDim int) int {
	return 3*inputDim*interDim + 2*interDim + inputDim
}

func packSwiGLUWeights(sl SwiGLULayerStream) ([]float32, error) {
	want := swigluWeightSize(sl.InputDim, sl.IntermediateDim)
	if len(sl.Weights) != want {
		return nil, fmt.Errorf("weights len %d != %d", len(sl.Weights), want)
	}
	out := make([]float32, want)
	for i, v := range sl.Weights {
		out[i] = float32(v)
	}
	return out, nil
}

func buildSwiGLULayer(sl SwiGLULayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "swiglu" {
		return fmt.Errorf("layer %d: want kind swiglu, got %q", layerIdx, sl.Kind)
	}
	w, err := packSwiGLUWeights(sl)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layerIdx, err)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerSwiGLU
	layer.Activation = poly.ActivationSilu
	layer.DType = poly.DTypeFloat32
	layer.InputHeight = sl.InputDim
	layer.OutputHeight = sl.IntermediateDim
	layer.WeightStore = poly.NewWeightStore(len(w))
	copy(layer.WeightStore.Master, w)
	return nil
}

// BuildNetworkFromSwiGLUStream builds a volumetric SwiGLU stack.
func BuildNetworkFromSwiGLUStream(req SwiGLUStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("swiglu stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("swiglu stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.InputDim != req.InputDim {
			return nil, fmt.Errorf("swiglu layer %d: input %d != expected %d", i, sl.InputDim, req.InputDim)
		}
		if sl.IntermediateDim != req.IntermediateDim {
			return nil, fmt.Errorf("swiglu layer %d: intermediate %d != expected %d", i, sl.IntermediateDim, req.IntermediateDim)
		}
		if err := buildSwiGLULayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamSwiGLUToEntity builds SwiGLU network, saves .entity, infers on fixture.
func StreamSwiGLUToEntity(req SwiGLUStreamRequest, modelsDir string, fx *SwiGLUFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromSwiGLUStream(req)
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
	xTest := SliceSwiGLUTestInputs(fx, seqLen, req.InputDim)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = seqLen * req.InputDim
	}
	outs := InferSwiGLUStack(net, xTest, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForSwiGLUStream returns streamed SwiGLU .entity path.
func EntityPathForSwiGLUStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
