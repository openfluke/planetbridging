package host

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"

	"github.com/openfluke/planetbridging/bridge"
)

func (s *Server) handleRMSNormStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var req bridge.RMSNormStreamRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Planet == "" || req.ModelID == "" {
		http.Error(w, "planet and model_id required", http.StatusBadRequest)
		return
	}
	if len(req.Layers) == 0 {
		http.Error(w, "layers required", http.StatusBadRequest)
		return
	}
	fixturesDir := filepath.Join("python", "rmsnorm", "fixtures")
	fx, err := bridge.LoadRMSNormFixture(req.FixtureVersion, fixturesDir)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, loomStreamResponse{
			Status:  "error",
			Message: fmt.Sprintf("fixture: %v", err),
		})
		return
	}
	result, err := bridge.StreamRMSNormToEntity(req, s.rmsnormModelsDir, fx)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, loomStreamResponse{
			Status:  "error",
			Message: err.Error(),
		})
		return
	}
	outDim := req.OutputDim
	if outDim == 0 && len(result.Outputs) > 0 {
		outDim = len(result.Outputs[0])
	}
	report := Report{
		Bedrock:        "rmsnorm",
		Planet:         req.Planet,
		Stage:          "loom",
		Format:         "entity",
		Engine:         req.Planet,
		ModelID:        req.ModelID,
		FixtureVersion: req.FixtureVersion,
		InputDim:       req.SeqLen * req.Dim,
		OutputDim:      outDim,
		SampleCount:    len(result.Outputs),
		Outputs:        result.Outputs,
		ArtifactPaths:  []string{result.EntityPath},
		TrainSkipped:   true,
	}
	if _, err := s.rmsnormStore.Save(report); err != nil {
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
	for _, rep := range s.allReports() {
		rep.Normalize()
		if rep.Bedrock == "rmsnorm" && rep.Planet == req.Planet && rep.ModelID == req.ModelID && rep.Stage == "native" {
			max, mean, exact := diffOutputs(rep.Outputs, result.Outputs)
			resp.MaxAbsDiff = max
			resp.MeanAbsDiff = mean
			resp.ExactMatch = exact
			break
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleCompareRMSNorm(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, CompareReports(filterBedrock(s.allReports(), "rmsnorm")))
}
