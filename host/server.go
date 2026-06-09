package host

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type Server struct {
	denseStore     *Store
	cnn1Store      *Store
	cnn2Store      *Store
	cnn3Store      *Store
	mhaStore       *Store
	lstmStore      *Store
	rnnStore       *Store
	denseModelsDir string
	cnn1ModelsDir  string
	cnn2ModelsDir  string
	cnn3ModelsDir  string
	mhaModelsDir   string
	lstmModelsDir  string
	rnnModelsDir   string
	mux            *http.ServeMux
}

func NewServer(
	denseStore, cnn1Store, cnn2Store, cnn3Store, mhaStore, lstmStore, rnnStore *Store,
	denseModelsDir, cnn1ModelsDir, cnn2ModelsDir, cnn3ModelsDir, mhaModelsDir, lstmModelsDir, rnnModelsDir string,
) *Server {
	s := &Server{
		denseStore:     denseStore,
		cnn1Store:      cnn1Store,
		cnn2Store:      cnn2Store,
		cnn3Store:      cnn3Store,
		mhaStore:       mhaStore,
		lstmStore:      lstmStore,
		rnnStore:       rnnStore,
		denseModelsDir: denseModelsDir,
		cnn1ModelsDir:  cnn1ModelsDir,
		cnn2ModelsDir:  cnn2ModelsDir,
		cnn3ModelsDir:  cnn3ModelsDir,
		mhaModelsDir:   mhaModelsDir,
		lstmModelsDir:  lstmModelsDir,
		rnnModelsDir:   rnnModelsDir,
		mux:            http.NewServeMux(),
	}
	s.routes()
	return s
}

func (s *Server) allReports() []Report {
	dense, _ := s.denseStore.LoadAll()
	for i := range dense {
		dense[i].Bedrock = "dense"
	}
	cnn1, _ := s.cnn1Store.LoadAll()
	for i := range cnn1 {
		cnn1[i].Bedrock = "cnn1"
	}
	cnn2, _ := s.cnn2Store.LoadAll()
	for i := range cnn2 {
		cnn2[i].Bedrock = "cnn2"
	}
	cnn3, _ := s.cnn3Store.LoadAll()
	for i := range cnn3 {
		cnn3[i].Bedrock = "cnn3"
	}
	mha, _ := s.mhaStore.LoadAll()
	for i := range mha {
		mha[i].Bedrock = "mha"
	}
	lstm, _ := s.lstmStore.LoadAll()
	for i := range lstm {
		lstm[i].Bedrock = "lstm"
	}
	rnn, _ := s.rnnStore.LoadAll()
	for i := range rnn {
		rnn[i].Bedrock = "rnn"
	}
	return append(append(append(append(append(append(dense, cnn1...), cnn2...), cnn3...), mha...), lstm...), rnn...)
}

func (s *Server) Handler() http.Handler { return s.mux }

func (s *Server) routes() {
	s.mux.HandleFunc("GET /", s.handleIndex)
	s.mux.HandleFunc("GET /PROGRESS.md", s.handleProgress)
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("POST /api/v1/report", s.handleReport)
	s.mux.HandleFunc("GET /api/v1/reports", s.handleReports)
	s.mux.HandleFunc("GET /api/v1/compare", s.handleCompare)
	s.mux.HandleFunc("GET /api/v1/compare/cnn1", s.handleCompareCNN1)
	s.mux.HandleFunc("GET /api/v1/compare/cnn2", s.handleCompareCNN2)
	s.mux.HandleFunc("GET /api/v1/compare/cnn3", s.handleCompareCNN3)
	s.mux.HandleFunc("GET /api/v1/compare/mha", s.handleCompareMHA)
	s.mux.HandleFunc("GET /api/v1/compare/lstm", s.handleCompareLSTM)
	s.mux.HandleFunc("GET /api/v1/compare/rnn", s.handleCompareRNN)
	s.mux.HandleFunc("GET /api/v1/compare.txt", s.handleCompareText)
	s.mux.HandleFunc("GET /api/v1/export/all.pdf", s.handleExportAllPDF)
	s.mux.HandleFunc("GET /api/v1/loom/catalog", s.handleLoomCatalog)
	s.mux.HandleFunc("POST /api/v1/loom/import", s.handleLoomImport)
	s.mux.HandleFunc("POST /api/v1/loom/stream", s.handleLoomImport)
	s.mux.HandleFunc("POST /api/v1/loom/stream/cnn1", s.handleCNN1Stream)
	s.mux.HandleFunc("POST /api/v1/loom/stream/cnn2", s.handleCNN2Stream)
	s.mux.HandleFunc("POST /api/v1/loom/stream/cnn3", s.handleCNN3Stream)
	s.mux.HandleFunc("POST /api/v1/loom/stream/mha", s.handleMHAStream)
	s.mux.HandleFunc("POST /api/v1/loom/stream/lstm", s.handleLSTMStream)
	s.mux.HandleFunc("POST /api/v1/loom/stream/rnn", s.handleRNNStream)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleReport(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 32<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var report Report
	if err := json.Unmarshal(body, &report); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if report.ModelID == "" {
		http.Error(w, "model_id required", http.StatusBadRequest)
		return
	}
	if report.Planet == "" && report.Engine == "" {
		http.Error(w, "planet and model_id required", http.StatusBadRequest)
		return
	}
	report.Normalize()
	if report.SampleCount == 0 {
		report.SampleCount = len(report.Outputs)
	}
	store := s.denseStore
	switch report.Bedrock {
	case "cnn1":
		store = s.cnn1Store
	case "cnn2":
		store = s.cnn2Store
	case "cnn3":
		store = s.cnn3Store
	case "mha":
		store = s.mhaStore
	case "lstm":
		store = s.lstmStore
	case "rnn":
		store = s.rnnStore
	}
	path, err := store.Save(report)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"saved": path})
}

func (s *Server) handleReports(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.allReports())
}

func (s *Server) handleCompare(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, CompareReports(filterBedrock(s.allReports(), "dense")))
}

func (s *Server) handleCompareCNN1(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, CompareReports(filterBedrock(s.allReports(), "cnn1")))
}

func (s *Server) handleCompareCNN2(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, CompareReports(filterBedrock(s.allReports(), "cnn2")))
}

func (s *Server) handleCompareCNN3(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, CompareReports(filterBedrock(s.allReports(), "cnn3")))
}

func (s *Server) handleCompareMHA(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, CompareReports(filterBedrock(s.allReports(), "mha")))
}

func (s *Server) handleCompareText(w http.ResponseWriter, r *http.Request) {
	bedrock := r.URL.Query().Get("bedrock")
	if bedrock == "" {
		bedrock = "dense"
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write([]byte(FormatBedrockComparisonText(bedrock, CompareReports(filterBedrock(s.allReports(), bedrock)))))
}

func (s *Server) handleExportAllPDF(w http.ResponseWriter, r *http.Request) {
	all := s.allReports()
	summaries := make(map[string]DenseComparisonSummary, len(AllBedrockIDs))
	for _, id := range AllBedrockIDs {
		summaries[id] = SortDenseSummary(CompareDensePipeline(filterBedrock(all, id)))
	}
	body, err := RenderAllComparisonPDF(Version, summaries)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/pdf")
	w.Header().Set("Content-Disposition", `attachment; filename="planetbridging-compare-all.pdf"`)
	_, _ = w.Write(body)
}

func filterBedrock(reports []Report, bedrock string) []Report {
	var out []Report
	for _, r := range reports {
		r.Normalize()
		if r.Bedrock == bedrock {
			out = append(out, r)
		}
	}
	return out
}

func (s *Server) handleLoomCatalog(w http.ResponseWriter, _ *http.Request) {
	dash, err := s.buildDashboard("dense")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"fixture": dash.Fixture,
		"dense":   dash.Dense,
		"models":  dash.LoomRows,
	})
}

func (s *Server) handleProgress(w http.ResponseWriter, _ *http.Request) {
	path := "PROGRESS.md"
	if wd, err := os.Getwd(); err == nil {
		if _, err := os.Stat(filepath.Join(wd, "PROGRESS.md")); err == nil {
			path = filepath.Join(wd, "PROGRESS.md")
		}
	}
	body, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write(body)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// HostReachable returns true when compare-host responds on baseURL.
func HostReachable(baseURL string) bool {
	baseURL = strings.TrimRight(baseURL, "/")
	resp, err := http.Get(baseURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
