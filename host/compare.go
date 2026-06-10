package host

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// FP32PassTolerance treats tiny float drift (ONNX, Loom entity infer) as pass in the UI.
const FP32PassTolerance = 1e-5

// PipelineStepDiff compares two steps in the same planet pipeline for one model.
type PipelineStepDiff struct {
	Planet      string  `json:"planet"`
	ModelID     string  `json:"model_id"`
	FromStage   string  `json:"from_stage"`
	FromFormat  string  `json:"from_format"`
	ToStage     string  `json:"to_stage"`
	ToFormat    string  `json:"to_format"`
	MaxAbsDiff  float64 `json:"max_abs_diff"`
	MeanAbsDiff float64 `json:"mean_abs_diff"`
	ExactMatch  bool    `json:"exact_match"`
	Pending     bool    `json:"pending,omitempty"`
}

// PlanetPipeline is native → export → loom for one training planet.
type PlanetPipeline struct {
	Planet  string             `json:"planet"`
	Steps   []Report           `json:"steps"`
	Compare []PipelineStepDiff `json:"compare"`
}

// DenseModelComparison is all pipelines for one bedrock model id.
type DenseModelComparison struct {
	ModelID   string           `json:"model_id"`
	Pipelines []PlanetPipeline `json:"pipelines"`
}

// DenseComparisonSummary is the dense-tab comparison (no cross-planet diffs).
type DenseComparisonSummary struct {
	FixtureVersion string                 `json:"fixture_version"`
	ReportCount    int                    `json:"report_count"`
	Models         []DenseModelComparison `json:"models"`
}

func CompareDensePipeline(reports []Report) DenseComparisonSummary {
	reports = filterActiveReports(reports)
	fixture := ""
	for i := range reports {
		if fixture == "" {
			fixture = reports[i].FixtureVersion
		}
	}

	byModelPlanet := map[string]map[string][]Report{}
	for _, r := range reports {
		if byModelPlanet[r.ModelID] == nil {
			byModelPlanet[r.ModelID] = map[string][]Report{}
		}
		byModelPlanet[r.ModelID][r.Planet] = append(byModelPlanet[r.ModelID][r.Planet], r)
	}

	var models []DenseModelComparison
	for modelID, planets := range byModelPlanet {
		var pipelines []PlanetPipeline
		for planet, steps := range planets {
			sort.Slice(steps, func(i, j int) bool {
				return stageOrder(steps[i].Stage) < stageOrder(steps[j].Stage)
			})
			pipelines = append(pipelines, PlanetPipeline{
				Planet:  planet,
				Steps:   steps,
				Compare: comparePlanetPipeline(modelID, planet, steps),
			})
		}
		sort.Slice(pipelines, func(i, j int) bool {
			return pipelines[i].Planet < pipelines[j].Planet
		})
		models = append(models, DenseModelComparison{
			ModelID:   modelID,
			Pipelines: pipelines,
		})
	}
	sort.Slice(models, func(i, j int) bool {
		return models[i].ModelID < models[j].ModelID
	})

	return DenseComparisonSummary{
		FixtureVersion: fixture,
		ReportCount:    len(reports),
		Models:         models,
	}
}

func stageOrder(stage string) int {
	switch stage {
	case "native":
		return 0
	case "export":
		return 1
	case "loom":
		return 2
	default:
		return 9
	}
}

func comparePlanetPipeline(modelID, planet string, steps []Report) []PipelineStepDiff {
	native, hasNative := findStep(steps, "native")
	if !hasNative {
		return nil
	}

	var diffs []PipelineStepDiff
	for _, target := range []struct {
		stage  string
		format string
	}{
		{"export", ""},
		{"loom", "loom"},
	} {
		if target.stage == "export" {
			for _, s := range steps {
				if s.Stage != "export" {
					continue
				}
				diffs = append(diffs, diffSteps(modelID, planet, native, s))
			}
			continue
		}
		loom, ok := findStep(steps, "loom")
		if ok {
			diffs = append(diffs, diffSteps(modelID, planet, native, loom))
		} else {
			diffs = append(diffs, PipelineStepDiff{
				Planet:     planet,
				ModelID:    modelID,
				FromStage:  native.Stage,
				FromFormat: native.Format,
				ToStage:    "loom",
				ToFormat:   "entity",
				Pending:    true,
			})
		}
	}
	sort.Slice(diffs, func(i, j int) bool {
		a, b := diffs[i], diffs[j]
		if a.ToStage != b.ToStage {
			return stageOrder(a.ToStage) < stageOrder(b.ToStage)
		}
		return a.ToFormat < b.ToFormat
	})
	return diffs
}

func findStep(steps []Report, stage string) (Report, bool) {
	for _, s := range steps {
		if s.Stage == stage {
			return s, true
		}
	}
	return Report{}, false
}

func diffSteps(modelID, planet string, from, to Report) PipelineStepDiff {
	maxDiff, meanDiff, exact := diffOutputs(from.Outputs, to.Outputs)
	return PipelineStepDiff{
		Planet:      planet,
		ModelID:     modelID,
		FromStage:   from.Stage,
		FromFormat:  from.Format,
		ToStage:     to.Stage,
		ToFormat:    to.Format,
		MaxAbsDiff:  maxDiff,
		MeanAbsDiff: meanDiff,
		ExactMatch:  exact,
	}
}

func diffOutputs(a, b [][]float64) (maxDiff, meanDiff float64, exact bool) {
	if len(a) != len(b) {
		return math.Inf(1), math.Inf(1), false
	}
	var sum float64
	var n int
	exact = true
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

// PipelineCompareLabel returns EXACT, PASS (within tolerance), DIFF, or PENDING.
func PipelineCompareLabel(d PipelineStepDiff) string {
	if d.Pending {
		return "PENDING"
	}
	if d.ExactMatch {
		return "EXACT"
	}
	if d.MaxAbsDiff < FP32PassTolerance {
		return "PASS"
	}
	return "DIFF"
}

// FormatDiffPlain renders an absolute diff as a fixed-point decimal (easier than 5.96e-07).
func FormatDiffPlain(v float64) string {
	if v == 0 {
		return "0"
	}
	abs := math.Abs(v)
	switch {
	case abs >= 1:
		return fmt.Sprintf("%.6f", v)
	case abs >= 0.0001:
		return fmt.Sprintf("%.8f", v)
	default:
		exp := math.Floor(math.Log10(abs))
		places := int(-exp) + 3
		if places > 15 {
			places = 15
		}
		return fmt.Sprintf("%.*f", places, v)
	}
}

// DiffScaleHint gives a plain-English sense of how big the offset is.
func DiffScaleHint(v float64) string {
	if v == 0 {
		return "identical"
	}
	abs := math.Abs(v)
	switch {
	case abs < 1e-6:
		return "~millionth — fp32 rounding noise"
	case abs < FP32PassTolerance:
		return "~hundred-thousandth — within PASS tolerance"
	case abs < 1e-3:
		return "~thousandth"
	case abs < 0.01:
		return "~hundredth (0.0X)"
	case abs < 0.1:
		return "~tenth (0.X)"
	case abs < 1:
		return "up to ~1.0"
	default:
		return "≥1 — large mismatch, investigate"
	}
}

// PipelineCompareClass returns a CSS class for the compare row.
func PipelineCompareClass(d PipelineStepDiff) string {
	switch PipelineCompareLabel(d) {
	case "PENDING":
		return "pending"
	case "EXACT", "PASS":
		return "exact"
	default:
		return "diff"
	}
}

// AllBedrockIDs is tab order for full export.
var AllBedrockIDs = []string{"dense", "cnn1", "cnn2", "cnn3", "mha", "lstm", "rnn", "layernorm", "embedding", "rmsnorm", "swiglu", "residual", "mixer"}

func FormatBedrockComparisonText(bedrock string, summary DenseComparisonSummary) string {
	var b fmtBuilder
	b.Printf("Planet Bridging %s pipeline comparison\n", bedrock)
	b.Printf("fixture=%s reports=%d\n\n", summary.FixtureVersion, summary.ReportCount)
	for _, m := range summary.Models {
		b.Printf("model %s\n", m.ModelID)
		for _, p := range m.Pipelines {
			b.Printf("  planet %s\n", p.Planet)
			for _, s := range p.Steps {
				b.Printf("    step %s/%s samples=%d\n", s.Stage, s.Format, s.SampleCount)
			}
			for _, d := range p.Compare {
				if d.Pending {
					b.Printf("    PENDING %s/%s → loom/entity (stream layers)\n", d.FromStage, d.FromFormat)
					continue
				}
				b.Printf("    %s %s/%s → %s/%s\n",
					PipelineCompareLabel(d), d.FromStage, d.FromFormat, d.ToStage, d.ToFormat)
				b.Printf("      max %.6e  ≈ %s  (%s)\n",
					d.MaxAbsDiff, FormatDiffPlain(d.MaxAbsDiff), DiffScaleHint(d.MaxAbsDiff))
				b.Printf("      mean %.6e  ≈ %s  (%s)\n",
					d.MeanAbsDiff, FormatDiffPlain(d.MeanAbsDiff), DiffScaleHint(d.MeanAbsDiff))
			}
		}
		b.Printf("\n")
	}
	return b.String()
}

type fmtBuilder struct {
	s string
}

func (f *fmtBuilder) Printf(format string, args ...any) {
	f.s += fmt.Sprintf(format, args...)
}

func (f *fmtBuilder) String() string { return f.s }

// CompareReports keeps the JSON API shape used by /api/v1/compare.
func CompareReports(reports []Report) DenseComparisonSummary {
	return CompareDensePipeline(reports)
}

func FormatComparisonText(summary DenseComparisonSummary) string {
	return FormatBedrockComparisonText("dense", summary)
}

// FormatAllComparisonText concatenates every bedrock tab into one log file.
func FormatAllComparisonText(version string, summaries map[string]DenseComparisonSummary) string {
	var b fmtBuilder
	b.Printf("Planet Bridging — full compare export\n")
	b.Printf("version=%s\n", version)
	for i, id := range AllBedrockIDs {
		s, ok := summaries[id]
		if !ok {
			continue
		}
		if i > 0 {
			b.Printf("\n")
		}
		b.Printf("%s\n", strings.Repeat("=", 78))
		b.Printf("BEDROCK: %s\n", id)
		b.Printf("%s\n", strings.Repeat("=", 78))
		b.Printf("\n")
		b.Printf("%s", FormatBedrockComparisonText(id, s))
	}
	return b.String()
}

