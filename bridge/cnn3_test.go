package bridge

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestCNN3OutputSpatial(t *testing.T) {
	if got := CNN3OutputSpatial(4, 4, 1, 0); got != 1 {
		t.Fatalf("expected out 1, got %d", got)
	}
}

func TestBuildAndInferCNN3SingleConv(t *testing.T) {
	req := CNN3StreamRequest{
		InputChannels: 1,
		Depth:         2,
		Height:        2,
		Width:         2,
		OutputDim:     2,
		Layers: []CNN3LayerStream{{
			Kind:        "cnn3",
			Index:       0,
			InChannels:  1,
			Filters:     2,
			InputDepth:  2,
			InputHeight: 2,
			InputWidth:  2,
			KernelSize:  2,
			Stride:      1,
			Padding:     0,
			Activation:  "linear",
			Weights:     make([]float64, 2*1*2*2*2),
		}},
	}
	// filter0 taps (0,0,0), filter1 taps (1,1,1)
	req.Layers[0].Weights[0] = 1
	req.Layers[0].Weights[15] = 1

	net, err := BuildNetworkFromCNN3Stream(req)
	if err != nil {
		t.Fatal(err)
	}
	if net.Layers[0].Type != poly.LayerCNN3 {
		t.Fatalf("expected CNN3 layer")
	}

	xTest := [][][][][]float64{
		{
			{
				{{1, 0}, {0, 0}},
				{{0, 0}, {0, 4}},
			},
		},
	}
	out := InferCNN3Stack(net, xTest, 2)
	if len(out) != 1 || len(out[0]) != 2 {
		t.Fatalf("bad output shape: %#v", out)
	}
	if out[0][0] != 1 || out[0][1] != 4 {
		t.Fatalf("unexpected values: %v", out[0])
	}
}
