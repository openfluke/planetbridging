package bridge

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN2OutputSpatial(t *testing.T) {
	if got := CNN2OutputSpatial(8, 8, 1, 0); got != 1 {
		t.Fatalf("expected out 1, got %d", got)
	}
	if got := CNN2OutputSpatial(8, 3, 1, 1); got != 8 {
		t.Fatalf("expected out 8, got %d", got)
	}
}

func TestBuildAndInferCNN2SingleConv(t *testing.T) {
	req := CNN2StreamRequest{
		InputChannels: 1,
		Height:        2,
		Width:         2,
		OutputDim:     2,
		Layers: []CNN2LayerStream{{
			Kind:        "cnn2",
			Index:       0,
			InChannels:  1,
			Filters:     2,
			InputHeight: 2,
			InputWidth:  2,
			KernelSize:  2,
			Stride:      1,
			Padding:     0,
			Activation:  "linear",
			Weights:     make([]float64, 2*1*2*2),
		}},
	}
	// filter0 taps top-left, filter1 taps bottom-right
	req.Layers[0].Weights[0] = 1
	req.Layers[0].Weights[7] = 1

	net, err := BuildNetworkFromCNN2Stream(req)
	if err != nil {
		t.Fatal(err)
	}
	if net.Layers[0].Type != poly.LayerCNN2 {
		t.Fatalf("expected CNN2 layer")
	}

	xTest := [][][][]float64{{{{1, 2}, {3, 4}}}}
	out := InferCNN2Stack(net, xTest, 2)
	if len(out) != 1 || len(out[0]) != 2 {
		t.Fatalf("bad output shape: %#v", out)
	}
	if out[0][0] != 1 || out[0][1] != 4 {
		t.Fatalf("unexpected values: %v", out[0])
	}
}
