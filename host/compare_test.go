package host

import (
	"strings"
	"testing"
)

func TestDiffOutputsExact(t *testing.T) {
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{1, 2}, {3, 4}}
	max, mean, exact := diffOutputs(a, b)
	if !exact || max != 0 || mean != 0 {
		t.Fatalf("expected exact match, got max=%v mean=%v exact=%v", max, mean, exact)
	}
}

func TestCompareDensePipelineNativeToExport(t *testing.T) {
	reports := []Report{
		{Planet: "tensorflow", Stage: "native", Format: "keras", ModelID: "m1", FixtureVersion: "v1", Outputs: [][]float64{{1, 2}}},
		{Planet: "tensorflow", Stage: "export", Format: "saved_model", ModelID: "m1", FixtureVersion: "v1", Outputs: [][]float64{{1, 2}}},
		{Planet: "pytorch", Stage: "native", Format: "pytorch", ModelID: "m1", FixtureVersion: "v1", Outputs: [][]float64{{9}}},
	}
	summary := CompareDensePipeline(reports)
	if len(summary.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(summary.Models))
	}
	if len(summary.Models[0].Pipelines) != 2 {
		t.Fatalf("expected 2 planet pipelines, got %d", len(summary.Models[0].Pipelines))
	}
	tf := summary.Models[0].Pipelines[0]
	if tf.Planet != "pytorch" && summary.Models[0].Pipelines[1].Planet == "pytorch" {
		tf = summary.Models[0].Pipelines[1]
	}
	for _, p := range summary.Models[0].Pipelines {
		if p.Planet == "tensorflow" {
			tf = p
		}
	}
	if len(tf.Compare) == 0 || !tf.Compare[0].ExactMatch {
		t.Fatalf("expected exact native→export for tensorflow")
	}
}

func TestFormatDiffPlain(t *testing.T) {
	got := FormatDiffPlain(5.960464e-07)
	if got != "0.0000005960" {
		t.Fatalf("expected plain decimal, got %q", got)
	}
	if DiffScaleHint(5.960464e-07) != "~millionth — fp32 rounding noise" {
		t.Fatalf("unexpected hint: %s", DiffScaleHint(5.960464e-07))
	}
}

func TestRenderAllComparisonPDF(t *testing.T) {
	summaries := map[string]DenseComparisonSummary{
		"dense": {FixtureVersion: "dense_v1", ReportCount: 1, Models: []DenseModelComparison{{ModelID: "m1"}}},
	}
	pdf, err := RenderAllComparisonPDF("0.5.0", summaries)
	if err != nil {
		t.Fatal(err)
	}
	if len(pdf) < 100 || pdf[0] != '%' || pdf[1] != 'P' {
		t.Fatalf("expected PDF header, got %d bytes", len(pdf))
	}
}

func TestFormatBedrockComparisonTextPlainDiff(t *testing.T) {
	summary := CompareDensePipeline([]Report{
		{Planet: "pytorch", Stage: "native", Format: "pytorch", ModelID: "m1", Outputs: [][]float64{{1}}},
		{Planet: "pytorch", Stage: "loom", Format: "entity", ModelID: "m1", Outputs: [][]float64{{1 + 5.96e-7}}},
	})
	out := FormatBedrockComparisonText("dense", summary)
	if !strings.Contains(out, "≈") || !strings.Contains(out, "millionth") {
		t.Fatalf("expected plain diff in export:\n%s", out)
	}
}

func TestFormatAllComparisonText(t *testing.T) {
	summaries := map[string]DenseComparisonSummary{
		"dense": {FixtureVersion: "dense_v1", ReportCount: 2, Models: []DenseModelComparison{{ModelID: "m1"}}},
		"cnn1":  {FixtureVersion: "cnn1_v1", ReportCount: 1},
	}
	out := FormatAllComparisonText("0.5.0", summaries)
	if !strings.Contains(out, "version=0.5.0") || !strings.Contains(out, "BEDROCK: dense") || !strings.Contains(out, "BEDROCK: cnn1") {
		t.Fatalf("missing expected sections:\n%s", out)
	}
}

func TestPipelineCompareLabelTolerance(t *testing.T) {
	label := PipelineCompareLabel(PipelineStepDiff{MaxAbsDiff: 5.96e-7})
	if label != "PASS" {
		t.Fatalf("expected PASS within tolerance, got %s", label)
	}
}

func TestCompareDensePipelineLoomPending(t *testing.T) {
	reports := []Report{
		{Planet: "pytorch", Stage: "native", Format: "pytorch", ModelID: "m1", Outputs: [][]float64{{1}}},
	}
	summary := CompareDensePipeline(reports)
	pending := false
	for _, d := range summary.Models[0].Pipelines[0].Compare {
		if d.Pending && d.ToStage == "loom" {
			pending = true
		}
	}
	if !pending {
		t.Fatalf("expected pending loom compare row")
	}
}
