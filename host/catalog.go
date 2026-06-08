package host

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// FixtureInfo documents the shared bedrock dataset all Python engines train on.
type FixtureInfo struct {
	Version      string `json:"version"`
	Seed         int    `json:"seed"`
	TrainSamples int    `json:"train_samples"`
	TestSamples  int    `json:"test_samples"`
	Note         string `json:"note"`
}

// SavedPlanetModel is one trained checkpoint on disk from a Python planet.
type SavedPlanetModel struct {
	Engine           string   `json:"engine"`
	ModelID          string   `json:"model_id"`
	FrameworkVersion string   `json:"framework_version"`
	Artifacts        []string `json:"artifacts"`
	EntityFiles      []string `json:"entity_files,omitempty"`
	HasReport        bool     `json:"has_report"`
	HasLoomReport    bool     `json:"has_loom_report"`
	Dir              string   `json:"dir"`
}

// LoomImportRow groups planet checkpoints for one bedrock model id.
type LoomImportRow struct {
	ModelID       string             `json:"model_id"`
	Planets       []SavedPlanetModel `json:"planets"`
	ReportEngines []string           `json:"report_engines"`
}

// Dashboard is all data for the dense layer compare UI.
type Dashboard struct {
	Fixture   FixtureInfo
	Dense     DenseComparisonSummary
	Loom      LoomDashboardStats
	LoomRows  []LoomImportRow
	ModelsDir string
}

func computeLoomStats(dense DenseComparisonSummary, rows []LoomImportRow) LoomDashboardStats {
	var stats LoomDashboardStats
	for _, row := range rows {
		for _, p := range row.Planets {
			stats.EntityFileCount += len(p.EntityFiles)
			if p.HasLoomReport {
				stats.LoomReportCount++
			}
		}
	}
	for _, m := range dense.Models {
		for _, pipe := range m.Pipelines {
			for _, d := range pipe.Compare {
				if d.Pending && d.ToStage == "loom" {
					stats.PendingLoomSteps++
				}
			}
		}
	}
	return stats
}

func DefaultFixtureInfo() FixtureInfo {
	return FixtureInfo{
		Version:      "dense_bedrock_v2",
		Seed:         42,
		TrainSamples: 5000,
		TestSamples:  100,
		Note: "Each planet trains on the same deterministic X/y, then we compare " +
			"native → export → loom/entity (live layer stream → .entity → Loom infer) on the same 100 test inputs.",
	}
}

// LoomDashboardStats summarizes entity stream progress for the UI header.
type LoomDashboardStats struct {
	EntityFileCount  int
	LoomReportCount  int
	PendingLoomSteps int
}

func ScanSavedModels(modelsDir string, reports []Report) ([]LoomImportRow, error) {
	if modelsDir == "" {
		return nil, nil
	}
	if _, err := os.Stat(modelsDir); err != nil {
		return nil, nil
	}

	reported := map[string]map[string]bool{}
	loomReported := map[string]map[string]bool{}
	for _, r := range reports {
		r.Normalize()
		if reported[r.ModelID] == nil {
			reported[r.ModelID] = map[string]bool{}
			loomReported[r.ModelID] = map[string]bool{}
		}
		reported[r.ModelID][r.Planet] = true
		if r.Stage == "loom" {
			loomReported[r.ModelID][r.Planet] = true
		}
	}

	byModel := map[string][]SavedPlanetModel{}
	engines, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil, err
	}
	for _, eng := range engines {
		if !eng.IsDir() {
			continue
		}
		planet := eng.Name()
		engineDir := filepath.Join(modelsDir, planet)
		models, err := os.ReadDir(engineDir)
		if err != nil {
			continue
		}
		for _, m := range models {
			if !m.IsDir() {
				continue
			}
			modelID := m.Name()
			dir := filepath.Join(engineDir, modelID)
			metaPath := filepath.Join(dir, "meta.json")
			b, err := os.ReadFile(metaPath)
			if err != nil {
				continue
			}
			var meta struct {
				FrameworkVersion string   `json:"framework_version"`
				Artifacts        []string `json:"artifacts"`
			}
			if err := json.Unmarshal(b, &meta); err != nil {
				continue
			}
			var entityFiles []string
			entries, err := os.ReadDir(dir)
			if err == nil {
				for _, ent := range entries {
					if !ent.IsDir() && strings.HasSuffix(ent.Name(), ".entity") {
						entityFiles = append(entityFiles, ent.Name())
					}
				}
				sort.Strings(entityFiles)
			}
			byModel[modelID] = append(byModel[modelID], SavedPlanetModel{
				Engine:           planet,
				ModelID:          modelID,
				FrameworkVersion: meta.FrameworkVersion,
				Artifacts:        meta.Artifacts,
				EntityFiles:      entityFiles,
				HasReport:        reported[modelID][planet],
				HasLoomReport:    loomReported[modelID][planet],
				Dir:              dir,
			})
		}
	}

	var rows []LoomImportRow
	for modelID, planets := range byModel {
		sort.Slice(planets, func(i, j int) bool {
			return planets[i].Engine < planets[j].Engine
		})
		var reps []string
		for p := range reported[modelID] {
			reps = append(reps, p)
		}
		sort.Strings(reps)
		rows = append(rows, LoomImportRow{
			ModelID:       modelID,
			Planets:       planets,
			ReportEngines: reps,
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].ModelID < rows[j].ModelID
	})
	return rows, nil
}

func (s *Server) buildDashboard() (Dashboard, error) {
	reports, err := s.store.LoadAll()
	if err != nil {
		return Dashboard{}, err
	}
	dense := SortDenseSummary(CompareDensePipeline(reports))
	fixture := DefaultFixtureInfo()
	if dense.FixtureVersion != "" {
		fixture.Version = dense.FixtureVersion
	}
	loomRows, err := ScanSavedModels(s.modelsDir, reports)
	if err != nil {
		return Dashboard{}, err
	}
	return Dashboard{
		Fixture:   fixture,
		Dense:     dense,
		Loom:      computeLoomStats(dense, loomRows),
		LoomRows:  loomRows,
		ModelsDir: s.modelsDir,
	}, nil
}

func SortDenseSummary(summary DenseComparisonSummary) DenseComparisonSummary {
	sort.Slice(summary.Models, func(i, j int) bool {
		return summary.Models[i].ModelID < summary.Models[j].ModelID
	})
	for i := range summary.Models {
		sort.Slice(summary.Models[i].Pipelines, func(a, b int) bool {
			return summary.Models[i].Pipelines[a].Planet < summary.Models[i].Pipelines[b].Planet
		})
	}
	return summary
}
