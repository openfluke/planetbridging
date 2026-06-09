package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// MHALayerStream is one MHA block from a planet runtime.
type MHALayerStream struct {
	Kind       string    `json:"kind"`
	Index      int       `json:"index"`
	DModel     int       `json:"d_model"`
	NumHeads   int       `json:"num_heads"`
	NumKVHeads int       `json:"num_kv_heads"`
	HeadDim    int       `json:"head_dim"`
	SeqLen     int       `json:"seq_len"`
	QWeights   []float64 `json:"q_weights"` // [qDim × dModel] row-major
	QBias      []float64 `json:"q_bias,omitempty"`
	KWeights   []float64 `json:"k_weights"` // [kvDim × dModel]
	KBias      []float64 `json:"k_bias,omitempty"`
	VWeights   []float64 `json:"v_weights"`
	VBias      []float64 `json:"v_bias,omitempty"`
	OWeights   []float64 `json:"o_weights"` // [dModel × qDim]
	OBias      []float64 `json:"o_bias,omitempty"`
}

// MHAStreamRequest is POST body for MHA bedrock layer stream.
type MHAStreamRequest struct {
	Bedrock        string           `json:"bedrock"`
	Planet         string           `json:"planet"`
	ModelID        string           `json:"model_id"`
	FixtureVersion string           `json:"fixture_version"`
	DModel         int              `json:"d_model"`
	SeqLen         int              `json:"seq_len"`
	OutputDim      int              `json:"output_dim"`
	Layers         []MHALayerStream `json:"layers"`
}

func mhaDims(sl MHALayerStream) (qDim, kvDim int) {
	qDim = sl.NumHeads * sl.HeadDim
	kvDim = sl.NumKVHeads * sl.HeadDim
	if sl.NumKVHeads == 0 {
		kvDim = qDim
	}
	return qDim, kvDim
}

func packMHAWeights(sl MHALayerStream) ([]float32, error) {
	qDim, kvDim := mhaDims(sl)
	d := sl.DModel
	wantQ := qDim * d
	wantKV := kvDim * d
	wantO := d * qDim
	if len(sl.QWeights) != wantQ {
		return nil, fmt.Errorf("q_weights len %d != %d×%d", len(sl.QWeights), qDim, d)
	}
	if len(sl.KWeights) != wantKV {
		return nil, fmt.Errorf("k_weights len %d != %d×%d", len(sl.KWeights), kvDim, d)
	}
	if len(sl.VWeights) != wantKV {
		return nil, fmt.Errorf("v_weights len %d != %d×%d", len(sl.VWeights), kvDim, d)
	}
	if len(sl.OWeights) != wantO {
		return nil, fmt.Errorf("o_weights len %d != %d×%d", len(sl.OWeights), d, qDim)
	}

	total := wantQ + wantKV + wantKV + wantO + qDim + kvDim + kvDim + d
	out := make([]float32, total)
	n := 0
	for _, v := range sl.QWeights {
		out[n] = float32(v)
		n++
	}
	for _, v := range sl.KWeights {
		out[n] = float32(v)
		n++
	}
	for _, v := range sl.VWeights {
		out[n] = float32(v)
		n++
	}
	for _, v := range sl.OWeights {
		out[n] = float32(v)
		n++
	}
	appendBias := func(b []float64, want int) {
		for i := 0; i < want; i++ {
			if i < len(b) {
				out[n] = float32(b[i])
			}
			n++
		}
	}
	appendBias(sl.QBias, qDim)
	appendBias(sl.KBias, kvDim)
	appendBias(sl.VBias, kvDim)
	appendBias(sl.OBias, d)
	return out, nil
}

func buildMHALayer(sl MHALayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "mha" {
		return fmt.Errorf("layer %d: want kind mha, got %q", layerIdx, sl.Kind)
	}
	qDim, _ := mhaDims(sl)
	w, err := packMHAWeights(sl)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layerIdx, err)
	}

	numKV := sl.NumKVHeads
	if numKV == 0 {
		numKV = sl.NumHeads
	}

	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerMultiHeadAttention
	layer.DType = poly.DTypeFloat32
	layer.DModel = sl.DModel
	layer.NumHeads = sl.NumHeads
	layer.NumKVHeads = numKV
	layer.HeadDim = sl.HeadDim
	layer.QueryDim = qDim
	layer.SeqLength = sl.SeqLen
	layer.InputHeight = sl.DModel
	layer.OutputHeight = sl.DModel
	layer.MaxSeqLen = sl.SeqLen
	if layer.MaxSeqLen < 8 {
		layer.MaxSeqLen = 8
	}
	layer.WeightStore = poly.NewWeightStore(len(w))
	copy(layer.WeightStore.Master, w)
	return nil
}

// BuildNetworkFromMHAStream builds a volumetric MHA stack.
func BuildNetworkFromMHAStream(req MHAStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("mha stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("mha stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.DModel != req.DModel {
			return nil, fmt.Errorf("mha layer %d: d_model %d != expected %d", i, sl.DModel, req.DModel)
		}
		if sl.SeqLen != req.SeqLen {
			return nil, fmt.Errorf("mha layer %d: seq_len %d != expected %d", i, sl.SeqLen, req.SeqLen)
		}
		if err := buildMHALayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamMHAToEntity builds MHA network, saves .entity, infers on fixture.
func StreamMHAToEntity(req MHAStreamRequest, modelsDir string, fx *MHAFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromMHAStream(req)
	if err != nil {
		return nil, err
	}

	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}

	lc, wb, _ := RoundTripEntity(entityPath)
	xTest := SliceMHATestInputs(fx, req.SeqLen, req.DModel)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = req.SeqLen * req.DModel
	}
	outs := InferMHAStack(net, xTest, outDim)

	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForMHAStream returns streamed MHA .entity path.
func EntityPathForMHAStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
