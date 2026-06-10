package bridge

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStreamEmbeddingRoundTrip(t *testing.T) {
	fixturesDir := filepath.Join("..", "python", "embedding", "fixtures")
	fx, err := LoadEmbeddingFixture("embedding_bedrock_v1", fixturesDir)
	if err != nil {
		t.Skipf("fixture missing: %v", err)
	}
	vocab, embedDim, seq := 16, 4, 4
	table := make([]float64, vocab*embedDim)
	for i := range table {
		table[i] = float64(i%vocab)*0.1 + float64(i%embedDim)*0.01
	}
	req := EmbeddingStreamRequest{
		Planet:         "test",
		ModelID:        "embedding_16_4_4",
		FixtureVersion: "embedding_bedrock_v1",
		VocabSize:      vocab,
		SeqLen:         seq,
		EmbedDim:       embedDim,
		OutputDim:      seq * embedDim,
		Layers: []EmbeddingLayerStream{{
			Kind:         "embedding",
			Index:        0,
			VocabSize:    vocab,
			EmbeddingDim: embedDim,
			Weights:      table,
		}},
	}
	dir := t.TempDir()
	result, err := StreamEmbeddingToEntity(req, dir, fx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(result.EntityPath); err != nil {
		t.Fatal(err)
	}
	if len(result.Outputs) != len(fx.XTest) {
		t.Fatalf("outputs %d != fixture samples %d", len(result.Outputs), len(fx.XTest))
	}
}
