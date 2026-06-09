package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferRNNStack runs Loom RNN forward per sample (batch=1).
func InferRNNStack(net *poly.VolumetricNetwork, xTest [][][]float64, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	seqLen := len(xTest[0])
	inputSize := len(xTest[0][0])
	hiddenSize := net.Layers[0].OutputHeight
	flatPer := seqLen * hiddenSize
	if outputDim > 0 && flatPer > outputDim {
		flatPer = outputDim
	}
	out := make([][]float64, len(xTest))
	for i, sample := range xTest {
		data := make([]float32, seqLen*inputSize)
		for t := 0; t < seqLen; t++ {
			for d := 0; d < inputSize; d++ {
				data[t*inputSize+d] = float32(sample[t][d])
			}
		}
		t := poly.NewTensorFromSlice(data, 1, seqLen, inputSize)
		for li := range net.Layers {
			layer := &net.Layers[li]
			if layer.IsDisabled || layer.Type != poly.LayerRNN {
				continue
			}
			_, post := poly.RNNForwardPolymorphic(layer, t)
			t = post
		}
		row := make([]float64, flatPer)
		for j := 0; j < flatPer && j < len(t.Data); j++ {
			row[j] = float64(t.Data[j])
		}
		out[i] = row
	}
	return out
}
