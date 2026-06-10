package bridge

import "testing"

func TestInferLayerNormShape(t *testing.T) {
	dim, seq := 4, 4
	req := LayerNormStreamRequest{
		Dim:       dim,
		SeqLen:    seq,
		OutputDim: seq * dim,
		Layers: []LayerNormLayerStream{{
			Kind:    "layernorm",
			Index:   0,
			Dim:     dim,
			Weights: make([]float64, 2*dim),
		}},
	}
	net, err := BuildNetworkFromLayerNormStream(req)
	if err != nil {
		t.Fatal(err)
	}
	xTest := [][][]float64{{{1, 2, 3, 4}, {4, 3, 2, 1}, {0.5, 1.5, 2.5, 3.5}, {-1, -2, -3, -4}}}
	out := InferLayerNormStack(net, xTest, seq*dim)
	if len(out) != 1 || len(out[0]) != seq*dim {
		t.Fatalf("bad output shape: %#v", out)
	}
}
