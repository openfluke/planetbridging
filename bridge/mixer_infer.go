package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferMixerStack runs the mixer pipeline in Loom (v1 or v2).
func InferMixerStack(net *poly.VolumetricNetwork, biases DenseBiases, xTest [][][][][]float64, tokenTest [][]float64, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	if outputDim == 0 {
		outputDim = MixerOutputDim
	}
	out := make([][]float64, len(xTest))
	for si, sample := range xTest {
		var tokens []float64
		if si < len(tokenTest) {
			tokens = tokenTest[si]
		}
		out[si] = inferMixerSample(net, biases, sample, tokens, outputDim)
	}
	return out
}

func inferMixerSample(net *poly.VolumetricNetwork, biases DenseBiases, sample [][][][]float64, tokens []float64, outputDim int) []float64 {
	batch := 1
	c, d, h, w := MixerVolumeC, MixerVolumeD, MixerVolumeH, MixerVolumeW
	data := make([]float32, c*d*h*w)
	for ch := 0; ch < c; ch++ {
		for dep := 0; dep < d; dep++ {
			for row := 0; row < h; row++ {
				for col := 0; col < w; col++ {
					idx := ch*d*h*w + dep*h*w + row*w + col
					if ch < len(sample) && dep < len(sample[ch]) && row < len(sample[ch][dep]) && col < len(sample[ch][dep][row]) {
						data[idx] = float32(sample[ch][dep][row][col])
					}
				}
			}
		}
	}
	t := poly.NewTensorFromSlice(data, batch, c, d, h, w)

	var skipAttn, skipMLP *poly.Tensor[float32]

	for li := range net.Layers {
		layer := &net.Layers[li]
		if layer.IsDisabled {
			continue
		}
		switch layer.Type {
		case poly.LayerCNN3:
			_, t = poly.CNN3ForwardPolymorphic(layer, t)
			t = flattenTensor(t)
		case poly.LayerDense:
			t = denseTensor2D(layer, t, biases[li])
		case poly.LayerCNN2:
			t = reshapeToCNN2(t, batch, 1, MixerCNN2H, MixerCNN2W)
			_, t = poly.CNN2ForwardPolymorphic(layer, t)
			t = flattenTensor(t)
		case poly.LayerCNN1:
			t = reshapeToCNN1(t, batch, 1, MixerCNN1Len)
			_, t = poly.CNN1ForwardPolymorphic(layer, t)
			t = flattenTensor(t)
		case poly.LayerEmbedding:
			seq := MixerEmbedSeq
			tokenData := make([]float32, seq)
			for i := 0; i < seq && i < len(tokens); i++ {
				tokenData[i] = float32(tokens[i])
			}
			t = poly.NewTensorFromSlice(tokenData, seq)
			_, t = poly.EmbeddingForwardPolymorphic(layer, t)
			t = reshapeToSeq(flattenTensor(t), batch, MixerEmbedSeq, MixerEmbedDim)
		case poly.LayerLayerNorm:
			_, t = poly.LayerNormForwardPolymorphic(layer, t)
			skipAttn = t.Clone()
		case poly.LayerMultiHeadAttention:
			t = reshapeToSeq(flattenTensor(t), batch, MixerMHASeq, MixerMHADModel)
			_, t = poly.MHAForwardPolymorphic(layer, t)
		case poly.LayerResidual:
			if skipAttn != nil {
				_, t = poly.ResidualForwardPolymorphic(layer, t, skipAttn)
				skipAttn = nil
			} else if skipMLP != nil {
				_, t = poly.ResidualForwardPolymorphic(layer, t, skipMLP)
				skipMLP = nil
			}
		case poly.LayerRMSNorm:
			_, t = poly.RMSNormForwardPolymorphic(layer, t)
			skipMLP = t.Clone()
		case poly.LayerSwiGLU:
			_, t = poly.SwiGLUForwardPolymorphic(layer, t)
			t = reshapeToSeq(flattenTensor(t), batch, MixerRecurrentSeq, MixerRecurrentIn)
		case poly.LayerRNN:
			_, t = poly.RNNForwardPolymorphic(layer, t)
		case poly.LayerLSTM:
			_, t = poly.LSTMForwardPolymorphic(layer, t)
		}
	}

	flat := flattenTensor(t)
	row := make([]float64, outputDim)
	for j := 0; j < outputDim && j < len(flat.Data); j++ {
		row[j] = float64(flat.Data[j])
	}
	return row
}

func denseTensor2D(layer *poly.VolumetricLayer, input *poly.Tensor[float32], bias []float32) *poly.Tensor[float32] {
	_, post := denseForwardWithBias(layer, input, bias)
	return post
}

func flattenTensor(t *poly.Tensor[float32]) *poly.Tensor[float32] {
	if t == nil || len(t.Shape) <= 2 {
		return t
	}
	batch := t.Shape[0]
	n := 1
	for _, d := range t.Shape[1:] {
		n *= d
	}
	out := make([]float32, batch*n)
	copy(out, t.Data[:batch*n])
	return poly.NewTensorFromSlice(out, batch, n)
}

func reshapeToCNN2(flat *poly.Tensor[float32], batch, channels, height, width int) *poly.Tensor[float32] {
	data := make([]float32, batch*channels*height*width)
	copy(data, flat.Data[:len(data)])
	return poly.NewTensorFromSlice(data, batch, channels, height, width)
}

func reshapeToCNN1(flat *poly.Tensor[float32], batch, channels, length int) *poly.Tensor[float32] {
	data := make([]float32, batch*channels*length)
	copy(data, flat.Data[:len(data)])
	return poly.NewTensorFromSlice(data, batch, channels, length)
}

func reshapeToSeq(flat *poly.Tensor[float32], batch, seqLen, dim int) *poly.Tensor[float32] {
	data := make([]float32, batch*seqLen*dim)
	copy(data, flat.Data[:len(data)])
	return poly.NewTensorFromSlice(data, batch, seqLen, dim)
}
