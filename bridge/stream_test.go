package bridge_test

import (
	"path/filepath"
	"testing"

	"github.com/openfluke/planetbridging/bridge"
)

func TestStreamToEntityRoundTrip(t *testing.T) {
	fx, err := bridge.LoadFixture("dense_bedrock_v2", filepath.Join("..", "python", "dense", "fixtures"))
	if err != nil {
		t.Skip(err)
	}

	// Minimal 32→16→4 MLP weights from pytorch safetensors via python would be ideal;
	// use synthetic stream to prove layer-by-layer entity path.
	req := bridge.StreamRequest{
		Planet:         "test",
		ModelID:        "synthetic",
		FixtureVersion: "dense_bedrock_v2",
		InputDim:       4,
		Layers: []bridge.LayerStream{
			{
				Index: 0, InputDim: 4, OutputDim: 2, Activation: "relu",
				Weights: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
				Bias:    []float64{0.01, 0.02},
			},
			{
				Index: 1, InputDim: 2, OutputDim: 1, Activation: "linear",
				Weights: []float64{1.0, 0.0},
				Bias:    []float64{0.5},
			},
		},
	}

	dir := t.TempDir()
	res, err := bridge.StreamToEntity(req, dir, fx)
	if err != nil {
		t.Fatal(err)
	}
	if res.LayerCount != 2 {
		t.Fatalf("layers %d", res.LayerCount)
	}
	if len(res.Outputs) != len(fx.XTest) {
		t.Fatalf("outputs %d want %d", len(res.Outputs), len(fx.XTest))
	}

	reloaded, biases, err := bridge.LoadEntity(res.EntityPath)
	if err != nil {
		t.Fatal(err)
	}
	outs2 := bridge.InferDenseMLP(reloaded, biases, bridge.SliceTestInputs(fx, 4))
	max, _, exact := bridge.MaxAbsOutputDiff(res.Outputs, outs2)
	if !exact {
		t.Fatalf("entity reload max diff %e", max)
	}
}
