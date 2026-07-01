package bridge

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

// CLIRequest is the stdin JSON envelope for loom-stream (no HTTP).
type CLIRequest struct {
	Bedrock        string          `json:"bedrock"`
	Root           string          `json:"root,omitempty"`
	ModelsDir      string          `json:"models_dir,omitempty"`
	FixturesDir    string          `json:"fixtures_dir,omitempty"`
	FixtureVersion string          `json:"fixture_version,omitempty"`
	OutputPath     string          `json:"output_path,omitempty"`
	Inputs         [][]float64     `json:"inputs,omitempty"`
	SkipInfer      bool            `json:"skip_infer,omitempty"`
	Payload        json.RawMessage `json:"payload,omitempty"`
}

// CLIResponse mirrors the compare-host loom stream JSON shape.
type CLIResponse struct {
	Status      string      `json:"status"`
	EntityPath  string      `json:"entity_path,omitempty"`
	LayerCount  int         `json:"layer_count,omitempty"`
	WeightBytes int         `json:"weight_bytes,omitempty"`
	OutputDim   int         `json:"output_dim,omitempty"`
	SampleCount int         `json:"sample_count,omitempty"`
	Outputs     [][]float64 `json:"outputs,omitempty"`
	MaxAbsDiff  float64     `json:"max_abs_diff,omitempty"`
	MeanAbsDiff float64     `json:"mean_abs_diff,omitempty"`
	ExactMatch  bool        `json:"exact_match,omitempty"`
	Message     string      `json:"message,omitempty"`
}

// StreamFromCLI reads a CLIRequest and returns a CLIResponse (dense + all bedrocks).
func StreamFromCLI(req CLIRequest, body []byte) CLIResponse {
	bedrock := strings.ToLower(strings.TrimSpace(req.Bedrock))
	if bedrock == "" {
		bedrock = detectBedrock(body)
	}
	if bedrock == "" {
		return CLIResponse{Status: "error", Message: "bedrock required (dense, cnn1, mha, …)"}
	}

	modelsDir := req.ModelsDir
	fixturesDir := req.FixturesDir
	if req.Root != "" {
		if modelsDir == "" {
			modelsDir = filepath.Join(req.Root, "python", bedrock, "models")
		}
		if fixturesDir == "" {
			fixturesDir = filepath.Join(req.Root, "python", bedrock, "fixtures")
		}
	}
	if modelsDir == "" && req.OutputPath == "" {
		return CLIResponse{Status: "error", Message: "models_dir or output_path required"}
	}

	payload := body
	if len(req.Payload) > 0 {
		payload = req.Payload
	}

	switch bedrock {
	case "dense":
		return streamDenseCLI(payload, fixturesDir, req.FixtureVersion, StreamEntityOptions{
			ModelsDir:  modelsDir,
			OutputPath: req.OutputPath,
			Inputs:     req.Inputs,
			SkipInfer:  req.SkipInfer,
		})
	case "cnn1":
		return streamCNN1CLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "cnn2":
		return streamCNN2CLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "cnn3":
		return streamCNN3CLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "mha":
		return streamMHACLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "lstm":
		return streamLSTMCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "rnn":
		return streamRNNCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "layernorm":
		return streamLayerNormCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "embedding":
		return streamEmbeddingCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "rmsnorm":
		return streamRMSNormCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "swiglu":
		return streamSwiGLUCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "residual":
		return streamResidualCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	case "mixer":
		return streamMixerCLI(payload, fixturesDir, req.FixtureVersion, modelsDir)
	default:
		return CLIResponse{Status: "error", Message: fmt.Sprintf("unknown bedrock %q", bedrock)}
	}
}

func detectBedrock(body []byte) string {
	var peek struct {
		Bedrock string `json:"bedrock"`
	}
	if err := json.Unmarshal(body, &peek); err == nil && peek.Bedrock != "" {
		return strings.ToLower(peek.Bedrock)
	}
	var densePeek struct {
		Layers []struct {
			Kind string `json:"kind"`
		} `json:"layers"`
	}
	if err := json.Unmarshal(body, &densePeek); err == nil && len(densePeek.Layers) > 0 {
		if densePeek.Layers[0].Kind == "" {
			return "dense"
		}
		return strings.ToLower(densePeek.Layers[0].Kind)
	}
	return ""
}

func streamDenseCLI(body []byte, fixturesDir, fixtureVersion string, opts StreamEntityOptions) CLIResponse {
	var req StreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	if req.Planet == "" || req.ModelID == "" {
		return CLIResponse{Status: "error", Message: "planet and model_id required"}
	}
	if len(req.Layers) == 0 {
		return CLIResponse{Status: "error", Message: "layers required"}
	}
	fv := fixtureVersion
	if fv == "" {
		fv = req.FixtureVersion
	}
	if fv != "" && fixturesDir != "" && !opts.SkipInfer && len(opts.Inputs) == 0 {
		fx, err := LoadFixture(fv, fixturesDir)
		if err != nil {
			return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
		}
		opts.Fixture = fx
	}
	result, err := StreamToEntityWithOptions(req, opts)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func cliResponseFromResult(result *StreamResult) CLIResponse {
	outDim := 0
	if len(result.Outputs) > 0 {
		outDim = len(result.Outputs[0])
	}
	return CLIResponse{
		Status:      "ok",
		EntityPath:  result.EntityPath,
		LayerCount:  result.LayerCount,
		WeightBytes: result.WeightBytes,
		OutputDim:   outDim,
		SampleCount: len(result.Outputs),
		Outputs:     result.Outputs,
	}
}

func fixtureVersionFrom(body []byte, override string) string {
	if override != "" {
		return override
	}
	var peek map[string]any
	if err := json.Unmarshal(body, &peek); err == nil {
		if v, ok := peek["fixture_version"].(string); ok {
			return v
		}
	}
	return ""
}

func streamCNN1CLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req CNN1StreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadCNN1Fixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamCNN1ToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamCNN2CLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req CNN2StreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadCNN2Fixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamCNN2ToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamCNN3CLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req CNN3StreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadCNN3Fixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamCNN3ToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamMHACLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req MHAStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadMHAFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamMHAToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamLSTMCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req LSTMStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadLSTMFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamLSTMToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamRNNCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req RNNStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadRNNFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamRNNToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamLayerNormCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req LayerNormStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadLayerNormFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamLayerNormToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamEmbeddingCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req EmbeddingStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadEmbeddingFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamEmbeddingToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamRMSNormCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req RMSNormStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadRMSNormFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamRMSNormToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamSwiGLUCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req SwiGLUStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadSwiGLUFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamSwiGLUToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamResidualCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req ResidualStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadResidualFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamResidualToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}

func streamMixerCLI(body []byte, fixturesDir, fixtureVersion, modelsDir string) CLIResponse {
	var req MixerStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	fv := fixtureVersionFrom(body, fixtureVersion)
	fx, err := LoadMixerFixture(fv, fixturesDir)
	if err != nil {
		return CLIResponse{Status: "error", Message: fmt.Sprintf("fixture: %v", err)}
	}
	result, err := StreamMixerToEntity(req, modelsDir, fx)
	if err != nil {
		return CLIResponse{Status: "error", Message: err.Error()}
	}
	return cliResponseFromResult(result)
}
