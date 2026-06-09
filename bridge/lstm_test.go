package bridge

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestPackLSTMWeights(t *testing.T) {
	inp, hid := 2, 2
	gate := lstmGateSize(inp, hid)
	sl := LSTMLayerStream{
		Kind:       "lstm",
		Index:      0,
		InputSize:  inp,
		HiddenSize: hid,
		SeqLen:     2,
		IWeights:   make([]float64, gate),
		FWeights:   make([]float64, gate),
		GWeights:   make([]float64, gate),
		OWeights:   make([]float64, gate),
	}
	w, err := packLSTMWeights(sl)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 4*gate {
		t.Fatalf("unexpected packed len %d", len(w))
	}
}

func TestBuildAndInferLSTMSimple(t *testing.T) {
	inp, hid, seq := 2, 2, 2
	gate := lstmGateSize(inp, hid)
	mkGate := func(ih0 float64) []float64 {
		g := make([]float64, gate)
		g[0] = ih0
		g[inp*hid+hid*hid] = 0.1 // bias
		return g
	}
	req := LSTMStreamRequest{
		InputSize:  inp,
		HiddenSize: hid,
		SeqLen:     seq,
		OutputDim:  seq * hid,
		Layers: []LSTMLayerStream{{
			Kind:       "lstm",
			Index:      0,
			InputSize:  inp,
			HiddenSize: hid,
			SeqLen:     seq,
			IWeights:   mkGate(1),
			FWeights:   mkGate(0),
			GWeights:   mkGate(1),
			OWeights:   mkGate(1),
		}},
	}
	net, err := BuildNetworkFromLSTMStream(req)
	if err != nil {
		t.Fatal(err)
	}
	if net.Layers[0].Type != poly.LayerLSTM {
		t.Fatalf("expected LSTM layer")
	}
	xTest := [][][]float64{{{1, 0}, {0, 1}}}
	out := InferLSTMStack(net, xTest, seq*hid)
	if len(out) != 1 || len(out[0]) != seq*hid {
		t.Fatalf("bad output shape: %#v", out)
	}
}
