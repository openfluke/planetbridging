package host

import "testing"

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
