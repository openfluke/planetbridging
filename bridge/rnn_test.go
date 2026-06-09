package bridge

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestPackRNNWeights(t *testing.T) {
	inp, hid := 2, 2
	want := rnnWeightSize(inp, hid)
	sl := RNNLayerStream{
		Kind:       "rnn",
		Index:      0,
		InputSize:  inp,
		HiddenSize: hid,
		SeqLen:     2,
		Weights:    make([]float64, want),
	}
	w, err := packRNNWeights(sl)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != want {
		t.Fatalf("unexpected packed len %d", len(w))
	}
}

func TestBuildAndInferRNNSimple(t *testing.T) {
	inp, hid, seq := 2, 2, 2
	want := rnnWeightSize(inp, hid)
	w := make([]float64, want)
	w[0] = 1.0
	w[inp*hid+hid*hid] = 0.1 // bias
	req := RNNStreamRequest{
		InputSize:  inp,
		HiddenSize: hid,
		SeqLen:     seq,
		OutputDim:  seq * hid,
		Layers: []RNNLayerStream{{
			Kind:       "rnn",
			Index:      0,
			InputSize:  inp,
			HiddenSize: hid,
			SeqLen:     seq,
			Weights:    w,
		}},
	}
	net, err := BuildNetworkFromRNNStream(req)
	if err != nil {
		t.Fatal(err)
	}
	if net.Layers[0].Type != poly.LayerRNN {
		t.Fatalf("expected RNN layer")
	}
	xTest := [][][]float64{{{1, 0}, {0, 1}}}
	out := InferRNNStack(net, xTest, seq*hid)
	if len(out) != 1 || len(out[0]) != seq*hid {
		t.Fatalf("bad output shape: %#v", out)
	}
}
