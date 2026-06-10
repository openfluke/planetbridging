package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferResidualStack runs Loom Residual forward per sample (output = main + skip).
func InferResidualStack(net *poly.VolumetricNetwork, mainTest, skipTest [][][]float64, outputDim int) [][]float64 {
	if len(mainTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	seqLen := len(mainTest[0])
	dim := len(mainTest[0][0])
	flatPer := seqLen * dim
	if outputDim > 0 && flatPer > outputDim {
		flatPer = outputDim
	}
	out := make([][]float64, len(mainTest))
	for i, mainSample := range mainTest {
		skipSample := skipTest[i]
		mainData := make([]float32, seqLen*dim)
		skipData := make([]float32, seqLen*dim)
		for t := 0; t < seqLen; t++ {
			for d := 0; d < dim; d++ {
				mainData[t*dim+d] = float32(mainSample[t][d])
				skipData[t*dim+d] = float32(skipSample[t][d])
			}
		}
		mainT := poly.NewTensorFromSlice(mainData, seqLen, dim)
		skipT := poly.NewTensorFromSlice(skipData, seqLen, dim)
		result := mainT
		for li := range net.Layers {
			layer := &net.Layers[li]
			if layer.IsDisabled || layer.Type != poly.LayerResidual {
				continue
			}
			_, post := poly.ResidualForwardPolymorphic(layer, mainT, skipT)
			result = post
		}
		row := make([]float64, flatPer)
		for j := 0; j < flatPer && j < len(result.Data); j++ {
			row[j] = float64(result.Data[j])
		}
		out[i] = row
	}
	return out
}
