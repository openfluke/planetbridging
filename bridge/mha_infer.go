package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferMHAStack runs Loom MHA forward per sample (batch=1) to avoid cross-batch KV coupling.
func InferMHAStack(net *poly.VolumetricNetwork, xTest [][][]float64, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	seqLen := len(xTest[0])
	dModel := len(xTest[0][0])
	flatPer := seqLen * dModel
	if outputDim > 0 && flatPer > outputDim {
		flatPer = outputDim
	}

	out := make([][]float64, len(xTest))
	for i, sample := range xTest {
		data := make([]float32, seqLen*dModel)
		for t := 0; t < seqLen; t++ {
			for d := 0; d < dModel; d++ {
				data[t*dModel+d] = float32(sample[t][d])
			}
		}
		t := poly.NewTensorFromSlice(data, 1, seqLen, dModel)
		for li := range net.Layers {
			layer := &net.Layers[li]
			if layer.IsDisabled || layer.Type != poly.LayerMultiHeadAttention {
				continue
			}
			_, post := poly.MHAForwardPolymorphic(layer, t)
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
