package bridge

import (
	"encoding/json"
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// MixerStreamRequest is POST body for the all-layers mixer bedrock.
type MixerStreamRequest struct {
	Bedrock        string            `json:"bedrock"`
	Planet         string            `json:"planet"`
	ModelID        string            `json:"model_id"`
	FixtureVersion string            `json:"fixture_version"`
	OutputDim      int               `json:"output_dim"`
	Layers         []json.RawMessage `json:"layers"`
}

func buildMixerDenseLayer(sl LayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	wantW := sl.OutputDim * sl.InputDim
	if len(sl.Weights) != wantW {
		return fmt.Errorf("dense layer %d: weights len %d != %d×%d", layerIdx, len(sl.Weights), sl.OutputDim, sl.InputDim)
	}
	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerDense
	layer.Activation = manifestActivation(sl.Activation)
	layer.DType = poly.DTypeFloat32
	layer.InputHeight = sl.InputDim
	layer.OutputHeight = sl.OutputDim
	layer.WeightStore = poly.NewWeightStore(wantW)
	for j, v := range sl.Weights {
		layer.WeightStore.Master[j] = float32(v)
	}
	return nil
}

func denseBiasesFromLayer(sl LayerStream, layerIdx int) ([]float32, bool) {
	if len(sl.Bias) == 0 {
		return nil, false
	}
	if len(sl.Bias) != sl.OutputDim {
		return nil, false
	}
	b := make([]float32, len(sl.Bias))
	for j, v := range sl.Bias {
		b[j] = float32(v)
	}
	return b, true
}

// BuildNetworkFromMixerStream builds a mixer stack (v1: 10 layers, v2: 16 layers).
func BuildNetworkFromMixerStream(req MixerStreamRequest) (*poly.VolumetricNetwork, DenseBiases, error) {
	want := MixerLayerCountForModel(req.ModelID)
	if len(req.Layers) != want {
		return nil, nil, fmt.Errorf("mixer stream: want %d layers for %s, got %d", want, req.ModelID, len(req.Layers))
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, want)
	biases := make(DenseBiases)

	for i, raw := range req.Layers {
		var head struct {
			Kind  string `json:"kind"`
			Index int    `json:"index"`
		}
		if err := json.Unmarshal(raw, &head); err != nil {
			return nil, nil, fmt.Errorf("mixer layer %d: %w", i, err)
		}
		if head.Index != i {
			return nil, nil, fmt.Errorf("mixer layer index %d != position %d", head.Index, i)
		}
		switch head.Kind {
		case "cnn3":
			var sl CNN3LayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildCNN3Layer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "cnn2":
			var sl CNN2LayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildCNN2Layer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "cnn1":
			var sl CNN1LayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildCNN1Layer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "dense":
			var sl LayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildMixerDenseLayer(sl, n, i); err != nil {
				return nil, nil, err
			}
			if b, ok := denseBiasesFromLayer(sl, i); ok {
				biases[i] = b
			}
		case "mha":
			var sl MHALayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildMHALayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "rnn":
			var sl RNNLayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildRNNLayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "lstm":
			var sl LSTMLayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildLSTMLayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "embedding":
			var sl EmbeddingLayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildEmbeddingLayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "layernorm":
			var sl LayerNormLayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildLayerNormLayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "rmsnorm":
			var sl RMSNormLayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildRMSNormLayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "swiglu":
			var sl SwiGLULayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildSwiGLULayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		case "residual":
			var sl ResidualLayerStream
			if err := json.Unmarshal(raw, &sl); err != nil {
				return nil, nil, err
			}
			if err := buildResidualLayer(sl, n, i); err != nil {
				return nil, nil, err
			}
		default:
			return nil, nil, fmt.Errorf("mixer layer %d: unknown kind %q", i, head.Kind)
		}
	}
	return n, biases, nil
}

// StreamMixerToEntity builds mixer network, saves .entity, infers on fixture.
func StreamMixerToEntity(req MixerStreamRequest, modelsDir string, fx *MixerFixture) (*StreamResult, error) {
	net, biases, err := BuildNetworkFromMixerStream(req)
	if err != nil {
		return nil, err
	}
	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, biases)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}
	lc, wb, _ := RoundTripEntity(entityPath)
	outDim := req.OutputDim
	if outDim == 0 {
		outDim = MixerOutputDim
	}
	outs := InferMixerStack(net, biases, fx.XTest, fx.TokenTest, outDim)
	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForMixerStream returns streamed mixer .entity path.
func EntityPathForMixerStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
