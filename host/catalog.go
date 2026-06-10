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

// Dashboard is all data for the compare UI (dense + cnn1/2/3 tabs).
type Dashboard struct {
	Tab           string
	Fixture       FixtureInfo
	CNN1Fixture   FixtureInfo
	CNN2Fixture   FixtureInfo
	CNN3Fixture   FixtureInfo
	MHAFixture    FixtureInfo
	LSTMFixture   FixtureInfo
	RNNFixture       FixtureInfo
	LayerNormFixture  FixtureInfo
	EmbeddingFixture  FixtureInfo
	MixerFixture      FixtureInfo
	Dense         DenseComparisonSummary
	CNN1          DenseComparisonSummary
	CNN2          DenseComparisonSummary
	CNN3          DenseComparisonSummary
	MHA           DenseComparisonSummary
	LSTM          DenseComparisonSummary
	RNN           DenseComparisonSummary
	LayerNorm     DenseComparisonSummary
	Embedding     DenseComparisonSummary
	Mixer         DenseComparisonSummary
	Loom          LoomDashboardStats
	CNN1Loom      LoomDashboardStats
	CNN2Loom      LoomDashboardStats
	CNN3Loom      LoomDashboardStats
	MHALoom       LoomDashboardStats
	LSTMLoom      LoomDashboardStats
	RNNLoom       LoomDashboardStats
	LayerNormLoom  LoomDashboardStats
	EmbeddingLoom  LoomDashboardStats
	MixerLoom      LoomDashboardStats
	LoomRows      []LoomImportRow
	CNN1LoomRows  []LoomImportRow
	CNN2LoomRows  []LoomImportRow
	CNN3LoomRows  []LoomImportRow
	MHALoomRows   []LoomImportRow
	LSTMLoomRows  []LoomImportRow
	RNNLoomRows       []LoomImportRow
	LayerNormLoomRows  []LoomImportRow
	EmbeddingLoomRows  []LoomImportRow
	MixerLoomRows      []LoomImportRow
	ModelsDir     string
	CNN1ModelsDir string
	CNN2ModelsDir string
	CNN3ModelsDir string
	MHAModelsDir  string
	LSTMModelsDir string
	RNNModelsDir       string
	LayerNormModelsDir  string
	EmbeddingModelsDir  string
	MixerModelsDir      string
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
	cnn3Rep := filterBedrock(all, "cnn3")
	mhaRep := filterBedrock(all, "mha")
	lstmRep := filterBedrock(all, "lstm")
	rnnRep := filterBedrock(all, "rnn")
	layernormRep := filterBedrock(all, "layernorm")
	embeddingRep := filterBedrock(all, "embedding")
	mixerRep := filterBedrock(all, "mixer")

	dense := SortDenseSummary(CompareDensePipeline(denseRep))
	cnn1 := SortDenseSummary(CompareDensePipeline(cnn1Rep))
	cnn2 := SortDenseSummary(CompareDensePipeline(cnn2Rep))
	cnn3 := SortDenseSummary(CompareDensePipeline(cnn3Rep))
	mha := SortDenseSummary(CompareDensePipeline(mhaRep))
	lstm := SortDenseSummary(CompareDensePipeline(lstmRep))
	rnn := SortDenseSummary(CompareDensePipeline(rnnRep))
	layernorm := SortDenseSummary(CompareDensePipeline(layernormRep))
	embedding := SortDenseSummary(CompareDensePipeline(embeddingRep))
	mixer := SortDenseSummary(CompareDensePipeline(mixerRep))

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
	cnn3Fixture := FixtureInfo{
		Version: "cnn3_bedrock_v1",
		Seed:    42,
		Note:    "3D conv bedrock — shared synthetic volumes (NCDHW), planets: pytorch · tensorflow · jax.",
	}
	if cnn3.FixtureVersion != "" {
		cnn3Fixture.Version = cnn3.FixtureVersion
	}
	mhaFixture := FixtureInfo{
		Version: "mha_bedrock_v1",
		Seed:    42,
		Note:    "MHA bedrock — causal + RoPE (Loom semantics), planets: pytorch · tensorflow · jax.",
	}
	if mha.FixtureVersion != "" {
		mhaFixture.Version = mha.FixtureVersion
	}
	lstmFixture := FixtureInfo{
		Version: "lstm_bedrock_v1",
		Seed:    42,
		Note:    "LSTM bedrock — Loom gate layout (i,f,g,o), planets: pytorch · tensorflow · jax.",
	}
	if lstm.FixtureVersion != "" {
		lstmFixture.Version = lstm.FixtureVersion
	}
	rnnFixture := FixtureInfo{
		Version: "rnn_bedrock_v1",
		Seed:    42,
		Note:    "RNN bedrock — tanh cell, zero initial hidden (Loom semantics), planets: pytorch · tensorflow · jax.",
	}
	if rnn.FixtureVersion != "" {
		rnnFixture.Version = rnn.FixtureVersion
	}
	layernormFixture := FixtureInfo{
		Version: "layernorm_bedrock_v1",
		Seed:    42,
		Note:    "LayerNorm bedrock — per-token gamma/beta (Loom semantics, eps=1e-5), planets: pytorch · tensorflow · jax.",
	}
	if layernorm.FixtureVersion != "" {
		layernormFixture.Version = layernorm.FixtureVersion
	}
	embeddingFixture := FixtureInfo{
		Version: "embedding_bedrock_v1",
		Seed:    42,
		Note:    "Embedding bedrock — token-id lookup (Loom semantics), planets: pytorch · tensorflow · jax.",
	}
	if embedding.FixtureVersion != "" {
		embeddingFixture.Version = embedding.FixtureVersion
	}
	mixerFixture := FixtureInfo{
		Version: "mixer_bedrock_v1",
		Seed:    42,
		Note:    "All-layers mixer bedrock — CNN3→Dense→CNN2→Dense→CNN1→Dense→MHA→RNN→LSTM→Dense, planets: pytorch · tensorflow · jax.",
	}
	if mixer.FixtureVersion != "" {
		mixerFixture.Version = mixer.FixtureVersion
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
	cnn3Rows, err := ScanSavedModels(s.cnn3ModelsDir, cnn3Rep)
	if err != nil {
		return Dashboard{}, err
	}
	mhaRows, err := ScanSavedModels(s.mhaModelsDir, mhaRep)
	if err != nil {
		return Dashboard{}, err
	}
	lstmRows, err := ScanSavedModels(s.lstmModelsDir, lstmRep)
	if err != nil {
		return Dashboard{}, err
	}
	rnnRows, err := ScanSavedModels(s.rnnModelsDir, rnnRep)
	if err != nil {
		return Dashboard{}, err
	}
	layernormRows, err := ScanSavedModels(s.layernormModelsDir, layernormRep)
	if err != nil {
		return Dashboard{}, err
	}
	embeddingRows, err := ScanSavedModels(s.embeddingModelsDir, embeddingRep)
	if err != nil {
		return Dashboard{}, err
	}
	mixerRows, err := ScanSavedModels(s.mixerModelsDir, mixerRep)
	if err != nil {
		return Dashboard{}, err
	}
	return Dashboard{
		Tab:           tab,
		Fixture:       fixture,
		CNN1Fixture:   cnn1Fixture,
		CNN2Fixture:   cnn2Fixture,
		CNN3Fixture:   cnn3Fixture,
		MHAFixture:    mhaFixture,
		LSTMFixture:   lstmFixture,
		RNNFixture:       rnnFixture,
		LayerNormFixture:  layernormFixture,
		EmbeddingFixture:  embeddingFixture,
		MixerFixture:      mixerFixture,
		Dense:         dense,
		CNN1:          cnn1,
		CNN2:          cnn2,
		CNN3:          cnn3,
		MHA:           mha,
		LSTM:          lstm,
		RNN:           rnn,
		LayerNorm:     layernorm,
		Embedding:     embedding,
		Mixer:         mixer,
		Loom:          computeLoomStats(dense, denseRows),
		CNN1Loom:      computeLoomStats(cnn1, cnn1Rows),
		CNN2Loom:      computeLoomStats(cnn2, cnn2Rows),
		CNN3Loom:      computeLoomStats(cnn3, cnn3Rows),
		MHALoom:       computeLoomStats(mha, mhaRows),
		LSTMLoom:      computeLoomStats(lstm, lstmRows),
		RNNLoom:       computeLoomStats(rnn, rnnRows),
		LayerNormLoom:  computeLoomStats(layernorm, layernormRows),
		EmbeddingLoom:  computeLoomStats(embedding, embeddingRows),
		MixerLoom:      computeLoomStats(mixer, mixerRows),
		LoomRows:      denseRows,
		CNN1LoomRows:  cnn1Rows,
		CNN2LoomRows:  cnn2Rows,
		CNN3LoomRows:  cnn3Rows,
		MHALoomRows:   mhaRows,
		LSTMLoomRows:  lstmRows,
		RNNLoomRows:       rnnRows,
		LayerNormLoomRows:  layernormRows,
		EmbeddingLoomRows:  embeddingRows,
		MixerLoomRows:      mixerRows,
		ModelsDir:     s.denseModelsDir,
		CNN1ModelsDir: s.cnn1ModelsDir,
		CNN2ModelsDir: s.cnn2ModelsDir,
		CNN3ModelsDir: s.cnn3ModelsDir,
		MHAModelsDir:  s.mhaModelsDir,
		LSTMModelsDir: s.lstmModelsDir,
		RNNModelsDir:       s.rnnModelsDir,
		LayerNormModelsDir:  s.layernormModelsDir,
		EmbeddingModelsDir:  s.embeddingModelsDir,
		MixerModelsDir:      s.mixerModelsDir,
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
