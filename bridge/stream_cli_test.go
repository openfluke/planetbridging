package bridge_test

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/openfluke/planetbridging/bridge"
)

func TestStreamFromCLISyntheticDense(t *testing.T) {
	root := filepath.Join("..")
	fixturesDir := filepath.Join(root, "python", "dense", "fixtures")
	dir := t.TempDir()

	payload := []byte(`{
		"planet": "test",
		"model_id": "cli_synthetic",
		"fixture_version": "dense_bedrock_v2",
		"input_dim": 4,
		"layers": [
			{
				"index": 0, "input_dim": 4, "output_dim": 2, "activation": "relu",
				"weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
				"bias": [0.01, 0.02]
			},
			{
				"index": 1, "input_dim": 2, "output_dim": 1, "activation": "linear",
				"weights": [1.0, 0.0],
				"bias": [0.5]
			}
		]
	}`)

	outPath := filepath.Join(dir, "cli.entity")
	resp := bridge.StreamFromCLI(bridge.CLIRequest{
		Bedrock:        "dense",
		FixturesDir:    fixturesDir,
		FixtureVersion: "dense_bedrock_v2",
		OutputPath:     outPath,
	}, payload)

	if resp.Status != "ok" {
		t.Fatalf("status %q msg %q", resp.Status, resp.Message)
	}
	if resp.EntityPath != outPath {
		t.Fatalf("entity path %q want %q", resp.EntityPath, outPath)
	}
	if resp.SampleCount != 100 {
		t.Fatalf("samples %d", resp.SampleCount)
	}

	b, _ := json.Marshal(resp)
	if len(b) == 0 {
		t.Fatal("empty response json")
	}
}
