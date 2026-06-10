package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferLayerNormStack runs Loom LayerNorm forward per sample (batch=seq_len rows).
func InferLayerNormStack(net *poly.VolumetricNetwork, xTest [][][]float64, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	seqLen := len(xTest[0])
	dim := len(xTest[0][0])
	flatPer := seqLen * dim
	if outputDim > 0 && flatPer > outputDim {
		flatPer = outputDim
	}
	out := make([][]float64, len(xTest))
	for i, sample := range xTest {
		data := make([]float32, seqLen*dim)
		for t := 0; t < seqLen; t++ {
			for d := 0; d < dim; d++ {
				data[t*dim+d] = float32(sample[t][d])
			}
		}
		// Loom LayerNorm normalizes each row of shape [seq_len, dim].
		t := poly.NewTensorFromSlice(data, seqLen, dim)
		for li := range net.Layers {
			layer := &net.Layers[li]
			if layer.IsDisabled || layer.Type != poly.LayerLayerNorm {
				continue
			}
			_, post := poly.LayerNormForwardPolymorphic(layer, t)
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
