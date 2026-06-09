package bridge

import (
	"fmt"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// CNN3OutputSpatial computes conv3d output spatial size (same as PyTorch).
func CNN3OutputSpatial(spatial, kernel, stride, padding int) int {
	return (spatial+2*padding-kernel)/stride + 1
}

// CNN3LayerStream is one Conv3d layer from a planet runtime.
type CNN3LayerStream struct {
	Kind         string    `json:"kind"`
	Index        int       `json:"index"`
	InChannels   int       `json:"in_channels"`
	Filters      int       `json:"filters"`
	InputDepth   int       `json:"input_depth"`
	InputHeight  int       `json:"input_height"`
	InputWidth   int       `json:"input_width"`
	KernelSize   int       `json:"kernel_size"`
	Stride       int       `json:"stride"`
	Padding      int       `json:"padding"`
	Activation   string    `json:"activation"`
	Weights      []float64 `json:"weights"` // [filters × in_channels × kD × kH × kW]
	Bias         []float64 `json:"bias,omitempty"`
}

// CNN3StreamRequest is POST body for CNN3 bedrock layer stream.
type CNN3StreamRequest struct {
	Bedrock        string            `json:"bedrock"`
	Planet         string            `json:"planet"`
	ModelID        string            `json:"model_id"`
	FixtureVersion string            `json:"fixture_version"`
	InputChannels  int               `json:"input_channels"`
	Depth          int               `json:"depth"`
	Height         int               `json:"height"`
	Width          int               `json:"width"`
	OutputDim      int               `json:"output_dim"`
	Layers         []CNN3LayerStream `json:"layers"`
}

func buildCNN3Layer(sl CNN3LayerStream, n *poly.VolumetricNetwork, layerIdx int) error {
	if sl.Kind != "" && sl.Kind != "cnn3" {
		return fmt.Errorf("layer %d: want kind cnn3, got %q", layerIdx, sl.Kind)
	}
	outD := CNN3OutputSpatial(sl.InputDepth, sl.KernelSize, sl.Stride, sl.Padding)
	outH := CNN3OutputSpatial(sl.InputHeight, sl.KernelSize, sl.Stride, sl.Padding)
	outW := CNN3OutputSpatial(sl.InputWidth, sl.KernelSize, sl.Stride, sl.Padding)
	wantW := sl.Filters * sl.InChannels * sl.KernelSize * sl.KernelSize * sl.KernelSize
	if len(sl.Weights) != wantW {
		return fmt.Errorf("layer %d: weights len %d != %d×%d×%d³",
			layerIdx, len(sl.Weights), sl.Filters, sl.InChannels, sl.KernelSize)
	}

	layer := n.GetLayer(0, 0, 0, layerIdx)
	layer.Type = poly.LayerCNN3
	layer.Activation = manifestActivation(sl.Activation)
	layer.DType = poly.DTypeFloat32
	layer.InputChannels = sl.InChannels
	layer.InputDepth = sl.InputDepth
	layer.InputHeight = sl.InputHeight
	layer.InputWidth = sl.InputWidth
	layer.Filters = sl.Filters
	layer.OutputDepth = outD
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

// BuildNetworkFromCNN3Stream builds a volumetric CNN3 stack.
func BuildNetworkFromCNN3Stream(req CNN3StreamRequest) (*poly.VolumetricNetwork, error) {
	if len(req.Layers) == 0 {
		return nil, fmt.Errorf("cnn3 stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))

	inCh := req.InputChannels
	depth := req.Depth
	height := req.Height
	width := req.Width
	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, fmt.Errorf("cnn3 stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.InChannels != inCh {
			return nil, fmt.Errorf("cnn3 layer %d: in_channels %d != expected %d", i, sl.InChannels, inCh)
		}
		if sl.InputDepth != depth {
			return nil, fmt.Errorf("cnn3 layer %d: input_depth %d != expected %d", i, sl.InputDepth, depth)
		}
		if sl.InputHeight != height {
			return nil, fmt.Errorf("cnn3 layer %d: input_height %d != expected %d", i, sl.InputHeight, height)
		}
		if sl.InputWidth != width {
			return nil, fmt.Errorf("cnn3 layer %d: input_width %d != expected %d", i, sl.InputWidth, width)
		}
		if err := buildCNN3Layer(sl, n, i); err != nil {
			return nil, err
		}
		depth = CNN3OutputSpatial(depth, sl.KernelSize, sl.Stride, sl.Padding)
		height = CNN3OutputSpatial(height, sl.KernelSize, sl.Stride, sl.Padding)
		width = CNN3OutputSpatial(width, sl.KernelSize, sl.Stride, sl.Padding)
		inCh = sl.Filters
	}
	return n, nil
}

// StreamCNN3ToEntity builds CNN3 network, saves .entity, infers on 5D fixture.
func StreamCNN3ToEntity(req CNN3StreamRequest, modelsDir string, fx *CNN3Fixture) (*StreamResult, error) {
	net, err := BuildNetworkFromCNN3Stream(req)
	if err != nil {
		return nil, err
	}

	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, nil)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}

	lc, wb, _ := RoundTripEntity(entityPath)
	xTest := SliceCNN3TestInputs(fx, req.InputChannels, req.Depth, req.Height, req.Width)
	outs := InferCNN3Stack(net, xTest, req.OutputDim)

	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForCNN3Stream returns streamed CNN3 .entity path.
func EntityPathForCNN3Stream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}
