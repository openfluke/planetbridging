package bridge

import (
	"github.com/openfluke/loom/poly"
)

// InferCNN3Stack runs Loom forward on a CNN3 stack; flattens final [B,C,D,H,W] to [B, outputDim].
func InferCNN3Stack(net *poly.VolumetricNetwork, xTest [][][][][]float64, outputDim int) [][]float64 {
	if len(xTest) == 0 || len(net.Layers) == 0 {
		return nil
	}
	batch := len(xTest)
	inC := net.Layers[0].InputChannels
	inD := net.Layers[0].InputDepth
	inH := net.Layers[0].InputHeight
	inW := net.Layers[0].InputWidth

	vol := inD * inH * inW
	data := make([]float32, batch*inC*vol)
	for b := 0; b < batch; b++ {
		for ch := 0; ch < inC; ch++ {
			for dep := 0; dep < inD; dep++ {
				for row := 0; row < inH; row++ {
					for col := 0; col < inW; col++ {
						idx := b*inC*vol + ch*vol + dep*inH*inW + row*inW + col
						data[idx] = float32(xTest[b][ch][dep][row][col])
					}
				}
			}
		}
	}
	t := poly.NewTensorFromSlice(data, batch, inC, inD, inH, inW)

	for i := range net.Layers {
		layer := &net.Layers[i]
		if layer.IsDisabled || layer.Type != poly.LayerCNN3 {
			continue
		}
		_, post := poly.CNN3ForwardPolymorphic(layer, t)
		t = post
	}

	flatPerBatch := 1
	for _, d := range t.Shape[1:] {
		flatPerBatch *= d
	}
	if outputDim > 0 && flatPerBatch > outputDim {
		flatPerBatch = outputDim
	}

	out := make([][]float64, batch)
	for b := 0; b < batch; b++ {
		row := make([]float64, flatPerBatch)
		base := b * flatPerBatch
		for j := 0; j < flatPerBatch && base+j < len(t.Data); j++ {
			row[j] = float64(t.Data[base+j])
		}
		if outputDim > 0 && len(row) > outputDim {
			row = row[:outputDim]
		}
		out[b] = row
	}
	return out
}
