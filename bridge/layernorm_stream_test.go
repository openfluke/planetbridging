package bridge

import "testing"

func TestStreamLayerNormToEntity(t *testing.T) {
	fx, err := LoadLayerNormFixture("layernorm_bedrock_v1", "../python/layernorm/fixtures")
	if err != nil {
		t.Fatal(err)
	}
	req := LayerNormStreamRequest{
		Bedrock:        "layernorm",
		Planet:         "test",
		ModelID:        "layernorm_4",
		FixtureVersion: "layernorm_bedrock_v1",
		Dim:            4,
		SeqLen:         4,
		OutputDim:      16,
		Layers: []LayerNormLayerStream{{
			Kind:    "layernorm",
			Index:   0,
			Dim:     4,
			Weights: []float64{1, 1, 1, 1, 0, 0, 0, 0},
		}},
	}
	dir := t.TempDir()
	result, err := StreamLayerNormToEntity(req, dir, fx)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Outputs) != 100 {
		t.Fatalf("want 100 samples, got %d", len(result.Outputs))
	}
}
