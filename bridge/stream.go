package bridge

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

// LayerStream is one dense layer streamed from a planet runtime (live weights, not a file).
type LayerStream struct {
	Index      int       `json:"index"`
	InputDim   int       `json:"input_dim"`
	OutputDim  int       `json:"output_dim"`
	Activation string    `json:"activation"`
	Weights    []float64 `json:"weights"`       // row-major [out × in], same layout as Loom Dense
	Bias       []float64 `json:"bias,omitempty"`
}

// StreamRequest is the POST body from Python after looping planet dense layers.
type StreamRequest struct {
	Planet         string        `json:"planet"`
	ModelID        string        `json:"model_id"`
	FixtureVersion string        `json:"fixture_version"`
	InputDim       int           `json:"input_dim"`
	Layers         []LayerStream `json:"layers"`
}

// StreamResult is entity path + Loom inference on the shared fixture.
type StreamResult struct {
	EntityPath  string
	Outputs     [][]float64
	LayerCount  int
	WeightBytes int
}

// BuildNetworkFromStream sets Loom volumetric dense layers one-by-one from streamed weights.
func BuildNetworkFromStream(req StreamRequest) (*poly.VolumetricNetwork, DenseBiases, error) {
	if len(req.Layers) == 0 {
		return nil, nil, fmt.Errorf("stream: no layers")
	}
	n := poly.NewVolumetricNetwork(1, 1, 1, len(req.Layers))
	biases := make(DenseBiases)
	inDim := req.InputDim

	for i, sl := range req.Layers {
		if sl.Index != i {
			return nil, nil, fmt.Errorf("stream: layer index %d != position %d", sl.Index, i)
		}
		if sl.InputDim != inDim {
			return nil, nil, fmt.Errorf("stream: layer %d input %d != expected %d", i, sl.InputDim, inDim)
		}
		wantW := sl.OutputDim * sl.InputDim
		if len(sl.Weights) != wantW {
			return nil, nil, fmt.Errorf("stream: layer %d weights len %d != %d×%d",
				i, len(sl.Weights), sl.OutputDim, sl.InputDim)
		}

		layer := n.GetLayer(0, 0, 0, i)
		layer.Type = poly.LayerDense
		layer.Activation = manifestActivation(sl.Activation)
		layer.DType = poly.DTypeFloat32
		layer.InputHeight = sl.InputDim
		layer.OutputHeight = sl.OutputDim
		layer.WeightStore = poly.NewWeightStore(wantW)
		for j, v := range sl.Weights {
			layer.WeightStore.Master[j] = float32(v)
		}

		if len(sl.Bias) > 0 {
			if len(sl.Bias) != sl.OutputDim {
				return nil, nil, fmt.Errorf("stream: layer %d bias len %d != %d", i, len(sl.Bias), sl.OutputDim)
			}
			b := make([]float32, len(sl.Bias))
			for j, v := range sl.Bias {
				b[j] = float32(v)
			}
			biases[i] = b
		}

		inDim = sl.OutputDim
	}
	return n, biases, nil
}

// StreamToEntity builds the network layer-by-layer, saves .entity, runs Loom infer.
func StreamToEntity(req StreamRequest, modelsDir string, fx *Fixture) (*StreamResult, error) {
	net, biases, err := BuildNetworkFromStream(req)
	if err != nil {
		return nil, err
	}

	entityPath, err := WriteEntityFromNetwork(modelsDir, req.Planet, req.ModelID, "stream", net, biases)
	if err != nil {
		return nil, fmt.Errorf("entity save: %w", err)
	}

	lc, wb, _ := RoundTripEntity(entityPath)
	xTest := SliceTestInputs(fx, req.InputDim)
	outs := InferDenseMLP(net, biases, xTest)

	return &StreamResult{
		EntityPath:  entityPath,
		Outputs:     outs,
		LayerCount:  lc,
		WeightBytes: wb,
	}, nil
}

// EntityPathForStream returns where streamed .entity files are written.
func EntityPathForStream(modelsDir, planet, modelID string) string {
	return filepath.Join(modelsDir, planet, modelID, modelID+".stream.entity")
}

// MaxAbsOutputDiff compares two output matrices.
func MaxAbsOutputDiff(a, b [][]float64) (maxDiff, meanDiff float64, exact bool) {
	if len(a) != len(b) {
		return math.Inf(1), math.Inf(1), false
	}
	exact = true
	var sum float64
	var n int
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return math.Inf(1), math.Inf(1), false
		}
		for j := range a[i] {
			d := math.Abs(a[i][j] - b[i][j])
			if d > maxDiff {
				maxDiff = d
			}
			sum += d
			n++
			if exact && a[i][j] != b[i][j] {
				exact = false
			}
		}
	}
	if n > 0 {
		meanDiff = sum / float64(n)
	}
	return maxDiff, meanDiff, exact
}
