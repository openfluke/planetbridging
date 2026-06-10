package bridge

import "testing"

func TestInferSwiGLUShape(t *testing.T) {
	inDim, interDim, seq := 4, 8, 4
	weights := make([]float64, swigluWeightSize(inDim, interDim))
	for i := range weights {
		weights[i] = float64(i%17) * 0.001
	}
	req := SwiGLUStreamRequest{
		InputDim:        inDim,
		IntermediateDim: interDim,
		SeqLen:          seq,
		OutputDim:       seq * inDim,
		Layers: []SwiGLULayerStream{{
			Kind:            "swiglu",
			Index:           0,
			InputDim:        inDim,
			IntermediateDim: interDim,
			Weights:         weights,
		}},
	}
	net, err := BuildNetworkFromSwiGLUStream(req)
	if err != nil {
		t.Fatal(err)
	}
	xTest := [][][]float64{{{0.1, 0.2, 0.3, 0.4}, {0.4, 0.3, 0.2, 0.1}, {-0.5, 0.5, -0.5, 0.5}, {1, -1, 1, -1}}}
	out := InferSwiGLUStack(net, xTest, seq*inDim)
	if len(out) != 1 || len(out[0]) != seq*inDim {
		t.Fatalf("bad output shape: %#v", out)
	}
}
