package host

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Report is one pipeline step forward pass on the shared 100-sample fixture.
type Report struct {
	Planet           string      `json:"planet"`
	Stage            string      `json:"stage"`
	Format           string      `json:"format"`
	Engine           string      `json:"engine"`
	ModelID          string      `json:"model_id"`
	FrameworkVersion string      `json:"framework_version"`
	FixtureVersion   string      `json:"fixture_version"`
	InputDim         int         `json:"input_dim"`
	OutputDim        int         `json:"output_dim"`
	SampleCount      int         `json:"sample_count"`
	Outputs          [][]float64 `json:"outputs"`
	ArtifactPaths    []string    `json:"artifact_paths,omitempty"`
	TrainSkipped     bool        `json:"train_skipped"`
	ReceivedAt       time.Time   `json:"received_at"`
}

func (r *Report) Normalize() {
	if r.Planet == "" {
		r.Planet = r.Engine
	}
	if r.Stage == "" {
		r.Stage = "native"
	}
	if r.Format == "" {
		r.Format = "native"
	}
	if r.Engine == "" {
		r.Engine = r.Planet
	}
}

type Store struct {
	dir string
	mu  sync.RWMutex
}

func NewStore(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

func (s *Store) Save(r Report) (string, error) {
	r.Normalize()
	if r.ReceivedAt.IsZero() {
		r.ReceivedAt = time.Now().UTC()
	}
	name := fmt.Sprintf(
		"%s__%s__%s__%s.json",
		sanitize(r.Planet),
		sanitize(r.ModelID),
		sanitize(r.Stage),
		sanitize(r.Format),
	)
	path := filepath.Join(s.dir, name)

	s.mu.Lock()
	defer s.mu.Unlock()

	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return "", err
	}
	return path, nil
}

func (s *Store) LoadAll() ([]Report, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}

	var out []Report
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		path := filepath.Join(s.dir, e.Name())
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		var r Report
		if err := json.Unmarshal(b, &r); err != nil {
			return nil, fmt.Errorf("%s: %w", e.Name(), err)
		}
		r.Normalize()
		out = append(out, r)
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].ModelID != out[j].ModelID {
			return out[i].ModelID < out[j].ModelID
		}
		if out[i].Planet != out[j].Planet {
			return out[i].Planet < out[j].Planet
		}
		if out[i].Stage != out[j].Stage {
			return out[i].Stage < out[j].Stage
		}
		return out[i].Format < out[j].Format
	})
	return out, nil
}

func sanitize(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "/", "_")
	return s
}

func reportKey(r Report) string {
	r.Normalize()
	return r.Planet + "\x00" + r.Stage + "\x00" + r.Format
}
