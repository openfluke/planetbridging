package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// ResidualLayerStream is one Residual layer from a planet runtime (no weights).
type ResidualLayerStream struct {
	Kind  string `json:"kind"`
	Index int    `json:"index"`
	Dim   int    `json:"dim"`
}

// ResidualStreamRequest is POST body for Residual bedrock layer stream.
type ResidualStreamRequest struct {
	Bedrock        string                `json:"bedrock"`
	Planet         string                `json:"planet"`
	ModelID        string                `json:"model_id"`
	FixtureVersion string                `json:"fixture_version"`
	Dim            int                   `json:"dim"`
	SeqLen         int                   `json:"seq_len"`
	OutputDim      int                   `json:"output_dim"`
	Layers         []ResidualLayerStream `json:"layers"`
}

func buildResidualLayer(sl ResidualLayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "residual" {
		return fmt.Errorf("layer %d: want kind residual, got %q", layerIdx, sl.Kind)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerResidual
	layer.Activation = poly.ActivationLinear
	layer.DType = poly.DTypeFloat32
	layer.InputHeight = sl.Dim
	layer.OutputHeight = sl.Dim
	return nil
}

// BuildNetworkFromResidualStream builds a volumetric Residual stack.
func BuildNetworkFromResidualStream(req ResidualStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("residual stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("residual stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.Dim != req.Dim {
			return nil, fmt.Errorf("residual layer %d: dim %d != expected %d", i, sl.Dim, req.Dim)
		}
		if err := buildResidualLayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamResidualToEntity builds Residual network, saves .entity, infers on fixture.
func StreamResidualToEntity(req ResidualStreamRequest, modelsDir string, fx *ResidualFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromResidualStream(req)
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
	mainTest := SliceResidualTestInputs(fx.MainTest, seqLen, req.Dim)
	skipTest := SliceResidualTestInputs(fx.SkipTest, seqLen, req.Dim)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = seqLen * req.Dim
	}
	outs := InferResidualStack(net, mainTest, skipTest, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForResidualStream returns streamed Residual .entity path.
func EntityPathForResidualStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
