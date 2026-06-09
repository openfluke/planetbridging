package bridge

import (
	"testing"

	"github.com/openfluke/loom/poly"
)

func TestPackMHAWeights(t *testing.T) {
	sl := MHALayerStream{
		Kind:       "mha",
		Index:      0,
		DModel:     4,
		NumHeads:     2,
		NumKVHeads: 2,
		HeadDim:    2,
		SeqLen:     2,
		QWeights:   make([]float64, 4*4),
		KWeights:   make([]float64, 4*4),
		VWeights:   make([]float64, 4*4),
		OWeights:   make([]float64, 4*4),
		QBias:      make([]float64, 4),
		KBias:      make([]float64, 4),
		VBias:      make([]float64, 4),
		OBias:      make([]float64, 4),
	}
	sl.QWeights[0] = 1
	sl.OWeights[0] = 1
	w, err := packMHAWeights(sl)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 4*4*4+4*4 {
		t.Fatalf("unexpected packed len %d", len(w))
	}
}

func TestBuildAndInferMHASingleHead(t *testing.T) {
	dModel := 2
	numHeads := 1
	headDim := 2
	seqLen := 2
	qDim := numHeads * headDim

	// Identity projections; zero bias -> output follows input through attention.
	qW := []float64{1, 0, 0, 1}
	kW := []float64{1, 0, 0, 1}
	vW := []float64{1, 0, 0, 1}
	oW := []float64{1, 0, 0, 1}

	req := MHAStreamRequest{
		DModel:    dModel,
		SeqLen:    seqLen,
		OutputDim: seqLen * dModel,
		Layers: []MHALayerStream{{
			Kind:       "mha",
			Index:      0,
			DModel:     dModel,
			NumHeads:   numHeads,
			NumKVHeads: numHeads,
			HeadDim:    headDim,
			SeqLen:     seqLen,
			QWeights:   qW,
			KWeights:   kW,
			VWeights:   vW,
			OWeights:   oW,
		}},
	}

	net, err := BuildNetworkFromMHAStream(req)
	if err != nil {
		t.Fatal(err)
	}
	if net.Layers[0].Type != poly.LayerMultiHeadAttention {
		t.Fatalf("expected MHA layer")
	}
	if net.Layers[0].QueryDim != qDim {
		t.Fatalf("query dim %d != %d", net.Layers[0].QueryDim, qDim)
	}

	xTest := [][][]float64{
		{{1, 0}, {0, 1}},
	}
	out := InferMHAStack(net, xTest, seqLen*dModel)
	if len(out) != 1 || len(out[0]) != seqLen*dModel {
		t.Fatalf("bad output shape: %#v", out)
	}
	if out[0][0] == 0 && out[0][1] == 0 {
		t.Fatalf("expected non-zero output, got %v", out[0])
	}
}
