package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// EmbeddingLayerStream is one Embedding layer from a planet runtime.
type EmbeddingLayerStream struct {
	Kind          string    `json:"kind"`
	Index         int       `json:"index"`
	VocabSize     int       `json:"vocab_size"`
	EmbeddingDim  int       `json:"embedding_dim"`
	Weights       []float64 `json:"weights"` // row-major [vocab × embed_dim]
}

// EmbeddingStreamRequest is POST body for Embedding bedrock layer stream.
type EmbeddingStreamRequest struct {
	Bedrock        string                 `json:"bedrock"`
	Planet         string                 `json:"planet"`
	ModelID        string                 `json:"model_id"`
	FixtureVersion string                 `json:"fixture_version"`
	VocabSize      int                    `json:"vocab_size"`
	SeqLen         int                    `json:"seq_len"`
	EmbedDim       int                    `json:"embed_dim"`
	OutputDim      int                    `json:"output_dim"`
	Layers         []EmbeddingLayerStream `json:"layers"`
}

func embeddingWeightSize(vocab, embedDim int) int {
	return vocab * embedDim
}

func packEmbeddingWeights(sl EmbeddingLayerStream) ([]float32, error) {
	want := embeddingWeightSize(sl.VocabSize, sl.EmbeddingDim)
	if len(sl.Weights) != want {
		return nil, fmt.Errorf("weights len %d != %d×%d", len(sl.Weights), sl.VocabSize, sl.EmbeddingDim)
	}
	out := make([]float32, want)
	for i, v := range sl.Weights {
		out[i] = float32(v)
	}
	return out, nil
}

func buildEmbeddingLayer(sl EmbeddingLayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "embedding" {
		return fmt.Errorf("layer %d: want kind embedding, got %q", layerIdx, sl.Kind)
	}
	w, err := packEmbeddingWeights(sl)
	if err != nil {
		return fmt.Errorf("layer %d: %w", layerIdx, err)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerEmbedding
	layer.Activation = poly.ActivationLinear
	layer.DType = poly.DTypeFloat32
	layer.VocabSize = sl.VocabSize
	layer.EmbeddingDim = sl.EmbeddingDim
	layer.InputHeight = sl.VocabSize
	layer.OutputHeight = sl.EmbeddingDim
	layer.WeightStore = poly.NewWeightStore(len(w))
	copy(layer.WeightStore.Master, w)
	return nil
}

// BuildNetworkFromEmbeddingStream builds a volumetric Embedding stack.
func BuildNetworkFromEmbeddingStream(req EmbeddingStreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("embedding stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("embedding stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.VocabSize != req.VocabSize {
			return nil, fmt.Errorf("embedding layer %d: vocab %d != expected %d", i, sl.VocabSize, req.VocabSize)
		}
		if sl.EmbeddingDim != req.EmbedDim {
			return nil, fmt.Errorf("embedding layer %d: embed_dim %d != expected %d", i, sl.EmbeddingDim, req.EmbedDim)
		}
		if err := buildEmbeddingLayer(sl, n, i); err != nil {
			return nil, err
		}
	}
	return n, nil
}

// StreamEmbeddingToEntity builds Embedding network, saves .entity, infers on fixture.
func StreamEmbeddingToEntity(req EmbeddingStreamRequest, modelsDir string, fx *EmbeddingFixture) (*StreamResult, error) {
	net, err := BuildNetworkFromEmbeddingStream(req)
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
	xTest := SliceEmbeddingTestInputs(fx, seqLen)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = seqLen * req.EmbedDim
	}
	outs := InferEmbeddingStack(net, xTest, seqLen, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForEmbeddingStream returns streamed Embedding .entity path.
func EntityPathForEmbeddingStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
