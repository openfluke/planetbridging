package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferCNN1Stack runs Loom forward on a CNN1 stack; flattens final [B,C,L] to [B, outputDim].
func InferCNN1Stack(net *poly.VolumetricNetwork, xTest [][][]float64, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	batch := len(xTest)
	inC := net.Layers[0].InputChannels
	seqLen := net.Layers[0].InputHeight

	data := make([]float32, batch*inC*seqLen)
	for b := 0; b < batch; b++ {
		for ch := 0; ch < inC; ch++ {
			for pos := 0; pos < seqLen; pos++ {
				data[b*inC*seqLen+ch*seqLen+pos] = float32(xTest[b][ch][pos])
			}
		}
	}
	t := poly.NewTensorFromSlice(data, batch, inC, seqLen)

	for i := range net.Layers {
		layer := &net.Layers[i]
		if layer.IsDisabled || layer.Type != poly.LayerCNN1 {
			continue
		}
		_, post := poly.CNN1ForwardPolymorphic(layer, t)
		t = post
	}

	// Flatten [batch, filters, outLen] → [batch, outputDim]
	total := 1
	for _, d := range t.Shape[1:] {
		total *= d
	}
	if outputDim > 0 && total > outputDim {
		total = outputDim
	}

	out := make([][]float64, batch)
	flatPerBatch := total
	if len(t.Shape) >= 3 {
		flatPerBatch = t.Shape[1] * t.Shape[2]
	} else if len(t.Shape) == 2 {
		flatPerBatch = t.Shape[1]
	}
	for b := 0; b < batch; b++ {
		row := make([]float64, flatPerBatch)
		base := b * flatPerBatch
		for j := 0; j < flatPerBatch && j < len(t.Data)-base; j++ {
			row[j] = float64(t.Data[base+j])
		}
		if outputDim > 0 && len(row) > outputDim {
			row = row[:outputDim]
		}
		out[b] = row
	}
	return out
}
