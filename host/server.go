package host

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

type Server struct {
	store     *Store
	modelsDir string
	mux       *http.ServeMux
}

func NewServer(store *Store, modelsDir string) *Server {
	s := &Server{store: store, modelsDir: modelsDir, mux: http.NewServeMux()}
	s.routes()
	return s
}

func (s *Server) Handler() http.Handler { return s.mux }

func (s *Server) routes() {
	s.mux.HandleFunc("GET /", s.handleIndex)
	s.mux.HandleFunc("GET /health", s.handleHealth)
	s.mux.HandleFunc("POST /api/v1/report", s.handleReport)
	s.mux.HandleFunc("GET /api/v1/reports", s.handleReports)
	s.mux.HandleFunc("GET /api/v1/compare", s.handleCompare)
	s.mux.HandleFunc("GET /api/v1/compare.txt", s.handleCompareText)
	s.mux.HandleFunc("GET /api/v1/loom/catalog", s.handleLoomCatalog)
	s.mux.HandleFunc("POST /api/v1/loom/import", s.handleLoomImport)
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
	path, err := s.store.Save(report)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"saved": path})
}

func (s *Server) handleReports(w http.ResponseWriter, _ *http.Request) {
	reports, err := s.store.LoadAll()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, reports)
}

func (s *Server) handleCompare(w http.ResponseWriter, _ *http.Request) {
	reports, err := s.store.LoadAll()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, CompareReports(reports))
}

func (s *Server) handleCompareText(w http.ResponseWriter, _ *http.Request) {
	reports, err := s.store.LoadAll()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write([]byte(FormatComparisonText(CompareReports(reports))))
}

func (s *Server) handleLoomCatalog(w http.ResponseWriter, _ *http.Request) {
	dash, err := s.buildDashboard()
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

func (s *Server) handleLoomImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusNotImplemented, map[string]string{
		"status":  "not_implemented",
		"message": "Loom dense import bridge is not wired yet. Planet outputs are saved for comparison once import lands.",
	})
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
