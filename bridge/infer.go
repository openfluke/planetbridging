package bridge

import "github.com/openfluke/loom/poly"

// InferDenseMLP runs Loom forward on a streamed dense stack (biases applied before activation).
func InferDenseMLP(net *poly.VolumetricNetwork, biases DenseBiases, xTest [][]float64) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	batch := len(xTest)
	inDim := net.Layers[0].InputHeight
	current := make([]float32, batch*inDim)
	for b := 0; b < batch; b++ {
		for i := 0; i < inDim; i++ {
			current[b*inDim+i] = float32(xTest[b][i])
		}
	}
	t := poly.NewTensorFromSlice(current, batch, inDim)

	for i := range net.Layers {
		layer := &net.Layers[i]
		if layer.IsDisabled || layer.Type != poly.LayerDense {
			continue
		}
		_, post := denseForwardWithBias(layer, t, biases[i])
		t = post
	}

	outDim := t.Shape[len(t.Shape)-1]
	out := make([][]float64, batch)
	for b := 0; b < batch; b++ {
		row := make([]float64, outDim)
		for o := 0; o < outDim; o++ {
			row[o] = float64(t.Data[b*outDim+o])
		}
		out[b] = row
	}
	return out
}

func denseForwardWithBias(layer *poly.VolumetricLayer, input *poly.Tensor[float32], bias []float32) (*poly.Tensor[float32], *poly.Tensor[float32]) {
	batch := input.Shape[0]
	inDim := layer.InputHeight
	outDim := layer.OutputHeight

	weights := layer.WeightStore.GetActive(layer.DType)
	if weights == nil {
		weights = layer.WeightStore.Master
	}
	w := poly.CastWeights[float32](weights)

	pre := poly.NewTensor[float32](batch, outDim)
	for b := 0; b < batch; b++ {
		for o := 0; o < outDim; o++ {
			var sum float64
			row := o * inDim
			for i := 0; i < inDim; i++ {
				sum += float64(input.Data[b*inDim+i]) * float64(w[row+i])
			}
			if len(bias) > 0 {
				sum += float64(bias[o])
			}
			pre.Data[b*outDim+o] = float32(sum)
		}
	}

	post := poly.NewTensor[float32](batch, outDim)
	for i := range post.Data {
		post.Data[i] = poly.Activate(pre.Data[i], layer.Activation)
	}
	return pre, post
}
