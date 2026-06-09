package bridge

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN1OutputLength(t *testing.T) {
	if got := CNN1OutputLength(32, 32, 1, 0); got != 1 {
		t.Fatalf("expected out len 1, got %d", got)
	}
	if got := CNN1OutputLength(32, 3, 1, 1); got != 32 {
		t.Fatalf("expected out len 32, got %d", got)
	}
}

func TestBuildAndInferCNN1SingleConv(t *testing.T) {
	req := CNN1StreamRequest{
		InputChannels: 1,
		SeqLen:        4,
		OutputDim:     2,
		Layers: []CNN1LayerStream{{
			Kind:        "cnn1",
			Index:       0,
			InChannels:  1,
			Filters:     2,
			InputLength: 4,
			KernelSize:  4,
			Stride:      1,
			Padding:     0,
			Activation:  "linear",
			Weights:     make([]float64, 2*1*4),
		}},
	}
	// identity-ish weights: filter0 taps x[0], filter1 taps x[3]
	req.Layers[0].Weights[0] = 1
	req.Layers[0].Weights[7] = 1

	net, err := BuildNetworkFromCNN1Stream(req)
	if err != nil {
		t.Fatal(err)
	}
	if net.Layers[0].Type != poly.LayerCNN1 {
		t.Fatalf("expected CNN1 layer")
	}

	xTest := [][][]float64{{{1, 2, 3, 4}}}
	out := InferCNN1Stack(net, xTest, 2)
	if len(out) != 1 || len(out[0]) != 2 {
		t.Fatalf("bad output shape: %#v", out)
	}
	if out[0][0] != 1 || out[0][1] != 4 {
		t.Fatalf("unexpected values: %v", out[0])
	}
}
