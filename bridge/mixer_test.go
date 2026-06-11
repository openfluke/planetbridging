package bridge

import (
	"encoding/json"
	"testing"
)

func TestInferMixerStackShape(t *testing.T) {
	req := minimalMixerRequest(t)
	net, biases, err := BuildNetworkFromMixerStream(req)
	if err != nil {
		t.Fatal(err)
	}
	sample := [][][][]float64{{
		{{0.1, 0.2}, {0.3, 0.4}},
		{{0.5, 0.6}, {0.7, 0.8}},
	}}
	out := InferMixerStack(net, biases, [][][][][]float64{sample}, nil, MixerOutputDim)
	if len(out) != 1 || len(out[0]) != MixerOutputDim {
		t.Fatalf("bad output shape: %#v", out)
	}
}

func minimalMixerRequest(t *testing.T) MixerStreamRequest {
	t.Helper()
	mkCNN3 := CNN3LayerStream{Kind: "cnn3", Index: 0, InChannels: 1, Filters: 2, InputDepth: 2, InputHeight: 2, InputWidth: 2, KernelSize: 2, Stride: 1, Padding: 0, Activation: "linear", Weights: make([]float64, 2*1*8)}
	mkDense := func(idx, in, out int, act string) map[string]any {
		return map[string]any{"kind": "dense", "index": idx, "input_dim": in, "output_dim": out, "activation": act, "weights": make([]float64, in*out)}
	}
	mkCNN2 := CNN2LayerStream{Kind: "cnn2", Index: 2, InChannels: 1, Filters: 2, InputHeight: 2, InputWidth: 4, KernelSize: 2, Stride: 1, Padding: 0, Activation: "linear", Weights: make([]float64, 2*1*4)}
	mkCNN1 := CNN1LayerStream{Kind: "cnn1", Index: 4, InChannels: 1, Filters: 4, InputLength: 8, KernelSize: 8, Stride: 1, Padding: 0, Activation: "linear", Weights: make([]float64, 4*1*8)}
	gate := lstmGateSize(MixerRecurrentIn, MixerRecurrentHid)
	mkLSTM := LSTMLayerStream{Kind: "lstm", Index: 8, InputSize: MixerRecurrentIn, HiddenSize: MixerRecurrentHid, SeqLen: MixerRecurrentSeq,
		IWeights: make([]float64, gate), FWeights: make([]float64, gate), GWeights: make([]float64, gate), OWeights: make([]float64, gate)}
	qDim := MixerMHAHeads * MixerMHAHeadDim
	kvDim := qDim
	d := MixerMHADModel
	mha := MHALayerStream{Kind: "mha", Index: 6, DModel: d, NumHeads: MixerMHAHeads, HeadDim: MixerMHAHeadDim, SeqLen: MixerMHASeq,
		QWeights: make([]float64, qDim*d), KWeights: make([]float64, kvDim*d), VWeights: make([]float64, kvDim*d), OWeights: make([]float64, d*qDim)}
	rnnW := rnnWeightSize(MixerRecurrentIn, MixerRecurrentHid)
	rnn := RNNLayerStream{Kind: "rnn", Index: 7, InputSize: MixerRecurrentIn, HiddenSize: MixerRecurrentHid, SeqLen: MixerRecurrentSeq, Weights: make([]float64, rnnW)}

	layers := []any{
		mkCNN3,
		mkDense(1, 2, 8, "linear"),
		mkCNN2,
		mkDense(3, 6, 8, "relu"),
		mkCNN1,
		mkDense(5, 4, 8, "linear"),
		mha,
		rnn,
		mkLSTM,
		mkDense(9, 8, 8, "linear"),
	}
	raws := make([]json.RawMessage, len(layers))
	for i, l := range layers {
		b, err := json.Marshal(l)
		if err != nil {
			t.Fatal(err)
		}
		raws[i] = b
	}
	return MixerStreamRequest{ModelID: MixerModelID, FixtureVersion: MixerFixtureVer, OutputDim: MixerOutputDim, Layers: raws}
}
