package bridge

import "testing"

func TestInferResidualShape(t *testing.T) {
	dim, seq := 4, 4
	req := ResidualStreamRequest{
		Dim:       dim,
		SeqLen:    seq,
		OutputDim: seq * dim,
		Layers: []ResidualLayerStream{{
			Kind:  "residual",
			Index: 0,
			Dim:   dim,
		}},
	}
	net, err := BuildNetworkFromResidualStream(req)
	if err != nil {
		t.Fatal(err)
	}
	main := [][][]float64{{{0.1, 0.2, 0.3, 0.4}, {0.4, 0.3, 0.2, 0.1}, {-0.5, 0.5, -0.5, 0.5}, {1, -1, 1, -1}}}
	skip := [][][]float64{{{0.01, 0.02, 0.03, 0.04}, {0.04, 0.03, 0.02, 0.01}, {-0.05, 0.05, -0.05, 0.05}, {0.1, -0.1, 0.1, -0.1}}}
	out := InferResidualStack(net, main, skip, seq*dim)
	if len(out) != 1 || len(out[0]) != seq*dim {
		t.Fatalf("bad output shape: %#v", out)
	}
	want0 := 0.1 + 0.01
	if out[0][0] < want0-1e-5 || out[0][0] > want0+1e-5 {
		t.Fatalf("bad first output: got %v want ~%v", out[0][0], want0)
	}
}
