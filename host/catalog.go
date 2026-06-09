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

// Dashboard is all data for the compare UI (dense + cnn1 + cnn2 tabs).
type Dashboard struct {
	Tab           string
	Fixture       FixtureInfo
	CNN1Fixture   FixtureInfo
	CNN2Fixture   FixtureInfo
	Dense         DenseComparisonSummary
	CNN1          DenseComparisonSummary
	CNN2          DenseComparisonSummary
	Loom          LoomDashboardStats
	CNN1Loom      LoomDashboardStats
	CNN2Loom      LoomDashboardStats
	LoomRows      []LoomImportRow
	CNN1LoomRows  []LoomImportRow
	CNN2LoomRows  []LoomImportRow
	ModelsDir     string
	CNN1ModelsDir string
	CNN2ModelsDir string
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
		if !isActivePlanet(planet) {
			continue
		}
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

func (s *Server) buildDashboard(tab string) (Dashboard, error) {
	if tab == "" {
		tab = "dense"
	}
	all := s.allReports()
	denseRep := filterBedrock(all, "dense")
	cnn1Rep := filterBedrock(all, "cnn1")
	cnn2Rep := filterBedrock(all, "cnn2")

	dense := SortDenseSummary(CompareDensePipeline(denseRep))
	cnn1 := SortDenseSummary(CompareDensePipeline(cnn1Rep))
	cnn2 := SortDenseSummary(CompareDensePipeline(cnn2Rep))

	fixture := DefaultFixtureInfo()
	if dense.FixtureVersion != "" {
		fixture.Version = dense.FixtureVersion
	}
	cnn1Fixture := FixtureInfo{
		Version: "cnn1_bedrock_v1",
		Seed:    42,
		Note:    "1D conv bedrock — shared synthetic sequences, planets: pytorch · tensorflow · jax.",
	}
	if cnn1.FixtureVersion != "" {
		cnn1Fixture.Version = cnn1.FixtureVersion
	}
	cnn2Fixture := FixtureInfo{
		Version: "cnn2_bedrock_v1",
		Seed:    42,
		Note:    "2D conv bedrock — shared synthetic grids (NCHW), planets: pytorch · tensorflow · jax.",
	}
	if cnn2.FixtureVersion != "" {
		cnn2Fixture.Version = cnn2.FixtureVersion
	}

	denseRows, err := ScanSavedModels(s.denseModelsDir, denseRep)
	if err != nil {
		return Dashboard{}, err
	}
	cnn1Rows, err := ScanSavedModels(s.cnn1ModelsDir, cnn1Rep)
	if err != nil {
		return Dashboard{}, err
	}
	cnn2Rows, err := ScanSavedModels(s.cnn2ModelsDir, cnn2Rep)
	if err != nil {
		return Dashboard{}, err
	}
	return Dashboard{
		Tab:           tab,
		Fixture:       fixture,
		CNN1Fixture:   cnn1Fixture,
		CNN2Fixture:   cnn2Fixture,
		Dense:         dense,
		CNN1:          cnn1,
		CNN2:          cnn2,
		Loom:          computeLoomStats(dense, denseRows),
		CNN1Loom:      computeLoomStats(cnn1, cnn1Rows),
		CNN2Loom:      computeLoomStats(cnn2, cnn2Rows),
		LoomRows:      denseRows,
		CNN1LoomRows:  cnn1Rows,
		CNN2LoomRows:  cnn2Rows,
		ModelsDir:     s.denseModelsDir,
		CNN1ModelsDir: s.cnn1ModelsDir,
		CNN2ModelsDir: s.cnn2ModelsDir,
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
