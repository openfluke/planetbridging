package bridge

import "testing"

func TestInferRMSNormShape(t *testing.T) {
	dim, seq := 4, 4
	gamma := make([]float64, dim)
	for i := range gamma {
		gamma[i] = 1.0
	}
	req := RMSNormStreamRequest{
		Dim:       dim,
		SeqLen:    seq,
		OutputDim: seq * dim,
		Layers: []RMSNormLayerStream{{
			Kind:    "rmsnorm",
			Index:   0,
			Dim:     dim,
			Weights: gamma,
		}},
	}
	net, err := BuildNetworkFromRMSNormStream(req)
	if err != nil {
		t.Fatal(err)
	}
	xTest := [][][]float64{{{1, 2, 3, 4}, {4, 3, 2, 1}, {0.5, 1.5, 2.5, 3.5}, {-1, -2, -3, -4}}}
	out := InferRMSNormStack(net, xTest, seq*dim)
	if len(out) != 1 || len(out[0]) != seq*dim {
		t.Fatalf("bad output shape: %#v", out)
	}
}
