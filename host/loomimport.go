package host

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"

	"github.com/openfluke/planetbridging/bridge"
)

type loomStreamResponse struct {
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

func (s *Server) handleLoomImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var req bridge.StreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Planet == "" || req.ModelID == "" {
		http.Error(w, "planet and model_id required", http.StatusBadRequest)
		return
	}
	if len(req.Layers) == 0 {
		http.Error(w, "layers required (stream dense weights from planet runtime)", http.StatusBadRequest)
		return
	}

	fixturesDir := filepath.Join("python", "dense", "fixtures")
	fx, err := bridge.LoadFixture(req.FixtureVersion, fixturesDir)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, loomStreamResponse{
			Status:  "error",
			Message: fmt.Sprintf("fixture: %v", err),
		})
		return
	}

	result, err := bridge.StreamToEntity(req, s.modelsDir, fx)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, loomStreamResponse{
			Status:  "error",
			Message: err.Error(),
		})
		return
	}

	outDim := 0
	if len(result.Outputs) > 0 {
		outDim = len(result.Outputs[0])
	}

	report := Report{
		Planet:         req.Planet,
		Stage:          "loom",
		Format:         "entity",
		Engine:         req.Planet,
		ModelID:        req.ModelID,
		FixtureVersion: req.FixtureVersion,
		InputDim:       req.InputDim,
		OutputDim:      outDim,
		SampleCount:    len(result.Outputs),
		Outputs:        result.Outputs,
		ArtifactPaths:  []string{result.EntityPath},
		TrainSkipped:   true,
	}
	if _, err := s.store.Save(report); err != nil {
		writeJSON(w, http.StatusInternalServerError, loomStreamResponse{
			Status:  "error",
			Message: fmt.Sprintf("report save: %v", err),
		})
		return
	}

	resp := loomStreamResponse{
		Status:      "ok",
		EntityPath:  result.EntityPath,
		LayerCount:  result.LayerCount,
		WeightBytes: result.WeightBytes,
		OutputDim:   outDim,
		SampleCount: len(result.Outputs),
		Outputs:     result.Outputs,
	}

	reports, _ := s.store.LoadAll()
	for _, rep := range reports {
		rep.Normalize()
		if rep.Planet == req.Planet && rep.ModelID == req.ModelID && rep.Stage == "native" {
			max, mean, exact := diffOutputs(rep.Outputs, result.Outputs)
			resp.MaxAbsDiff = max
			resp.MeanAbsDiff = mean
			resp.ExactMatch = exact
			break
		}
	}

	writeJSON(w, http.StatusOK, resp)
}
