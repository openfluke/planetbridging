package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferSwiGLUStack runs Loom SwiGLU forward per sample (batch=seq_len rows).
func InferSwiGLUStack(net *poly.VolumetricNetwork, xTest [][][]float64, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	seqLen := len(xTest[0])
	inputDim := len(xTest[0][0])
	flatPer := seqLen * inputDim
	if outputDim > 0 && flatPer > outputDim {
		flatPer = outputDim
	}
	out := make([][]float64, len(xTest))
	for i, sample := range xTest {
		data := make([]float32, seqLen*inputDim)
		for t := 0; t < seqLen; t++ {
			for d := 0; d < inputDim; d++ {
				data[t*inputDim+d] = float32(sample[t][d])
			}
		}
		t := poly.NewTensorFromSlice(data, seqLen, inputDim)
		for li := range net.Layers {
			layer := &net.Layers[li]
			if layer.IsDisabled || layer.Type != poly.LayerSwiGLU {
				continue
			}
			_, post := poly.SwiGLUForwardPolymorphic(layer, t)
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
