package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// CNN2OutputSpatial computes conv2d output spatial size (same as PyTorch).
func CNN2OutputSpatial(spatial, kernel, stride, padding int) int {
	return (spatial+2*padding-kernel)/stride + 1
}

// CNN2LayerStream is one Conv2d layer from a planet runtime.
type CNN2LayerStream struct {
	Kind         string    `json:"kind"`
	Index        int       `json:"index"`
	InChannels   int       `json:"in_channels"`
	Filters      int       `json:"filters"`
	InputHeight  int       `json:"input_height"`
	InputWidth   int       `json:"input_width"`
	KernelSize   int       `json:"kernel_size"`
	Stride       int       `json:"stride"`
	Padding      int       `json:"padding"`
	Activation   string    `json:"activation"`
	Weights      []float64 `json:"weights"` // [filters × in_channels × kH × kW]
	Bias         []float64 `json:"bias,omitempty"`
}

// CNN2StreamRequest is POST body for CNN2 bedrock layer stream.
type CNN2StreamRequest struct {
	Bedrock        string            `json:"bedrock"`
	Planet         string            `json:"planet"`
	ModelID        string            `json:"model_id"`
	FixtureVersion string            `json:"fixture_version"`
	InputChannels  int               `json:"input_channels"`
	Height         int               `json:"height"`
	Width          int               `json:"width"`
	OutputDim      int               `json:"output_dim"`
	Layers         []CNN2LayerStream `json:"layers"`
}

func buildCNN2Layer(sl CNN2LayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "cnn2" {
		return fmt.Errorf("layer %d: want kind cnn2, got %q", layerIdx, sl.Kind)
	}
	outH := CNN2OutputSpatial(sl.InputHeight, sl.KernelSize, sl.Stride, sl.Padding)
	outW := CNN2OutputSpatial(sl.InputWidth, sl.KernelSize, sl.Stride, sl.Padding)
	wantW := sl.Filters * sl.InChannels * sl.KernelSize * sl.KernelSize
	if len(sl.Weights) != wantW {
		return fmt.Errorf("layer %d: weights len %d != %d×%d×%d²",
			layerIdx, len(sl.Weights), sl.Filters, sl.InChannels, sl.KernelSize)
	}

	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerCNN2
	layer.Activation = manifestActivation(sl.Activation)
	layer.DType = poly.DTypeFloat32
	layer.InputChannels = sl.InChannels
	layer.InputHeight = sl.InputHeight
	layer.InputWidth = sl.InputWidth
	layer.Filters = sl.Filters
	layer.OutputHeight = outH
	layer.OutputWidth = outW
	layer.KernelSize = sl.KernelSize
	layer.Stride = sl.Stride
	layer.Padding = sl.Padding
	layer.WeightStore = poly.NewWeightStore(wantW)
	for j, v := range sl.Weights {
		layer.WeightStore.Master[j] = float32(v)
	}
	return nil
}

// BuildNetworkFromCNN2Stream builds a volumetric CNN2 stack.
func BuildNetworkFromCNN2Stream(req CNN2StreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("cnn2 stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))

	inCh := req.InputChannels
	height := req.Height
	width := req.Width
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("cnn2 stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.InChannels != inCh {
			return nil, fmt.Errorf("cnn2 layer %d: in_channels %d != expected %d", i, sl.InChannels, inCh)
		}
		if sl.InputHeight != height {
			return nil, fmt.Errorf("cnn2 layer %d: input_height %d != expected %d", i, sl.InputHeight, height)
		}
		if sl.InputWidth != width {
			return nil, fmt.Errorf("cnn2 layer %d: input_width %d != expected %d", i, sl.InputWidth, width)
		}
		if err := buildCNN2Layer(sl, n, i); err != nil {
			return nil, err
		}
		height = CNN2OutputSpatial(height, sl.KernelSize, sl.Stride, sl.Padding)
		width = CNN2OutputSpatial(width, sl.KernelSize, sl.Stride, sl.Padding)
		inCh = sl.Filters
	}
	return n, nil
}

// StreamCNN2ToEntity builds CNN2 network, saves .entity, infers on 4D fixture.
func StreamCNN2ToEntity(req CNN2StreamRequest, modelsDir string, fx *CNN2Fixture) (*StreamResult, error) {
	net, err := BuildNetworkFromCNN2Stream(req)
	if err != nil {
		return nil, err
	}

	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}

	lc, wb, _ := RoundTripEntity(entityPath)
	xTest := SliceCNN2TestInputs(fx, req.InputChannels, req.Height, req.Width)
	outs := InferCNN2Stack(net, xTest, req.OutputDim)

	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForCNN2Stream returns streamed CNN2 .entity path.
func EntityPathForCNN2Stream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
