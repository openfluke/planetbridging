package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferEmbeddingStack runs Loom Embedding forward per token-id sample.
func InferEmbeddingStack(net *poly.VolumetricNetwork, xTest [][]float64, seqLen, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	flatPer := seqLen * net.Layers[0].EmbeddingDim
	if outputDim > 0 && flatPer > outputDim {
		flatPer = outputDim
	}
	out := make([][]float64, len(xTest))
	for i, sample := range xTest {
		data := make([]float32, seqLen)
		for t := 0; t < seqLen; t++ {
			if t < len(sample) {
				data[t] = float32(sample[t])
			}
		}
		t := poly.NewTensorFromSlice(data, seqLen)
		for li := range net.Layers {
			layer := &net.Layers[li]
			if layer.IsDisabled || layer.Type != poly.LayerEmbedding {
				continue
			}
			_, post := poly.EmbeddingForwardPolymorphic(layer, t)
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
