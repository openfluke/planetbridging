package bridge

import "testing"

func TestInferEmbeddingShape(t *testing.T) {
	vocab, embedDim, seq := 8, 4, 4
	table := make([]float64, vocab*embedDim)
	for i := range table {
		table[i] = float64(i) * 0.01
	}
	req := EmbeddingStreamRequest{
		VocabSize: vocab,
		SeqLen:    seq,
		EmbedDim:  embedDim,
		OutputDim: seq * embedDim,
		Layers: []EmbeddingLayerStream{{
			Kind:         "embedding",
			Index:        0,
			VocabSize:    vocab,
			EmbeddingDim: embedDim,
			Weights:      table,
		}},
	}
	net, err := BuildNetworkFromEmbeddingStream(req)
	if err != nil {
		t.Fatal(err)
	}
	xTest := [][]float64{{0, 1, 2, 3}}
	out := InferEmbeddingStack(net, xTest, seq, seq*embedDim)
	if len(out) != 1 || len(out[0]) != seq*embedDim {
		t.Fatalf("bad output shape: %#v", out)
	}
}
