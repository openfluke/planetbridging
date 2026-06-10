package host

import (
	"bytes"
	"fmt"
	"html/template"
	"net/http"
)

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	tab := r.URL.Query().Get("tab")
	dash, err := s.buildDashboard(tab)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var buf bytes.Buffer
	if err := dashboardTmpl.Execute(&buf, dash); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(buf.Bytes())
}

var dashboardTmpl = template.Must(template.New("dashboard").Funcs(template.FuncMap{
	"compareClass": func(d PipelineStepDiff) string {
		return PipelineCompareClass(d)
	},
	"compareLabel": func(d PipelineStepDiff) string {
		return PipelineCompareLabel(d)
	},
	"fmtSci": func(v float64) string {
		return fmt.Sprintf("%.6e", v)
	},
	"fmtDiffPlain": FormatDiffPlain,
	"fmtDiffHint":  DiffScaleHint,
	"stepLabel": func(stage, format string) string {
		if stage == format {
			return stage
		}
		return stage + " / " + format
	},
	"stepClass": func(stage string) string {
		if stage == "loom" {
			return "loom"
		}
		return ""
	},
	"toLabel": func(d PipelineStepDiff) string {
		if d.Pending {
			return "loom / entity (stream pending)"
		}
		return d.ToStage + " / " + d.ToFormat
	},
	"fp32Tol": func() string {
		return fmt.Sprintf("%.0e", FP32PassTolerance)
	},
}).Parse(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Planet Bridging — Dense</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0d0d0d;
    --panel: #141414;
    --border: #2a2a2a;
    --text: #d4d4d4;
    --muted: #888;
    --accent: #7ec8e3;
    --loom: #ffb74d;
    --exact: #4caf50;
    --diff: #ffb74d;
    --pending: #666;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    padding: 24px 32px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  h1 { margin: 0 0 8px; font-size: 18px; color: var(--accent); font-weight: 600; }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
  .fixture-banner {
    padding: 10px 14px;
    border: 1px solid #1e3a4a;
    border-radius: 6px;
    background: rgba(126, 200, 227, 0.06);
    font-size: 12px;
  }
  .pipeline-hint {
    margin-top: 10px;
    color: var(--muted);
    font-size: 12px;
  }
  main { padding: 24px 32px 48px; max-width: 1400px; }
  .empty {
    border: 1px dashed var(--border);
    padding: 32px;
    color: var(--muted);
    border-radius: 8px;
  }
  .empty code { color: var(--accent); }
  section.model {
    margin-bottom: 28px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: var(--panel);
  }
  section.model h2 {
    margin: 0;
    padding: 14px 18px;
    font-size: 14px;
    border-bottom: 1px solid var(--border);
    background: #101010;
  }
  .planet-block {
    border-top: 1px solid var(--border);
  }
  .planet-block:first-child { border-top: none; }
  .planet-head {
    padding: 10px 18px;
    background: #111;
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .steps {
    padding: 8px 18px 4px;
    color: var(--muted);
    font-size: 11px;
  }
  .loom-stats {
    margin-top: 10px;
    padding: 8px 12px;
    border: 1px solid #3d2e1a;
    border-radius: 6px;
    background: rgba(255, 183, 77, 0.06);
    font-size: 12px;
    color: var(--loom);
  }
  .steps span {
    display: inline-block;
    margin-right: 12px;
    color: var(--text);
  }
  .steps span.loom { color: var(--loom); }
  .steps .artifact {
    display: block;
    margin: 4px 0 6px;
    color: var(--muted);
    font-size: 10px;
    word-break: break-all;
  }
  section.loom-catalog {
    margin-bottom: 28px;
    border: 1px solid #3d2e1a;
    border-radius: 8px;
    overflow: hidden;
    background: var(--panel);
  }
  section.loom-catalog h2 {
    margin: 0;
    padding: 14px 18px;
    font-size: 14px;
    border-bottom: 1px solid var(--border);
    background: #15120e;
    color: var(--loom);
  }
  .loom-catalog table td.mono { font-size: 11px; color: var(--muted); }
  .loom-yes { color: var(--exact); }
  .loom-no { color: var(--muted); }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th, td {
    padding: 8px 14px;
    text-align: left;
    border-top: 1px solid var(--border);
  }
  th { color: var(--muted); font-weight: 500; }
  tr.exact td { background: rgba(76, 175, 80, 0.08); }
  tr.diff td { background: rgba(255, 183, 77, 0.05); }
  tr.pending td { background: rgba(102, 102, 102, 0.08); color: var(--muted); }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge.exact { background: rgba(76,175,80,.2); color: var(--exact); }
  .badge.diff { background: rgba(255,183,77,.15); color: var(--diff); }
  .badge.pending { background: rgba(102,102,102,.2); color: #aaa; }
  .diff-num { line-height: 1.35; }
  .diff-sci { color: var(--text); }
  .diff-plain { color: var(--accent); font-size: 11px; }
  .diff-hint { color: var(--muted); font-size: 10px; font-style: italic; }
  footer {
    padding: 16px 32px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
  }
  a { color: var(--accent); }
  .tabs { margin: 12px 0 0; }
  .tabs a {
    margin-right: 16px;
    text-decoration: none;
    color: var(--muted);
    font-weight: 600;
  }
  .tabs a.active { color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 2px; }
  .header-actions {
    margin-top: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .export-btn {
    display: inline-block;
    padding: 6px 12px;
    border: 1px solid #1e3a4a;
    border-radius: 6px;
    background: rgba(126, 200, 227, 0.12);
    color: var(--accent);
    text-decoration: none;
    font-size: 12px;
    font-weight: 600;
  }
  .export-btn:hover { background: rgba(126, 200, 227, 0.22); }
  .export-hint { color: var(--muted); font-size: 11px; }
</style>
</head>
<body>
<header>
  <div class="tabs">
    <a href="/?tab=dense" {{if eq .Tab "dense"}}class="active"{{end}}>Dense</a>
    <a href="/?tab=cnn1" {{if eq .Tab "cnn1"}}class="active"{{end}}>CNN1</a>
    <a href="/?tab=cnn2" {{if eq .Tab "cnn2"}}class="active"{{end}}>CNN2</a>
    <a href="/?tab=cnn3" {{if eq .Tab "cnn3"}}class="active"{{end}}>CNN3</a>
    <a href="/?tab=mha" {{if eq .Tab "mha"}}class="active"{{end}}>MHA</a>
    <a href="/?tab=lstm" {{if eq .Tab "lstm"}}class="active"{{end}}>LSTM</a>
    <a href="/?tab=rnn" {{if eq .Tab "rnn"}}class="active"{{end}}>RNN</a>
    <a href="/?tab=layernorm" {{if eq .Tab "layernorm"}}class="active"{{end}}>LayerNorm</a>
    <a href="/?tab=embedding" {{if eq .Tab "embedding"}}class="active"{{end}}>Embedding</a>
    <a href="/?tab=mixer" {{if eq .Tab "mixer"}}class="active"{{end}}>Mixer</a>
  </div>
  <div class="header-actions">
    <a href="/api/v1/export/all.pdf" class="export-btn" download="planetbridging-compare-all.pdf">Export all tabs → .pdf</a>
    <span class="export-hint">dense · cnn1 · cnn2 · cnn3 · mha · lstm · rnn · layernorm · embedding · mixer — all models &amp; compare rows</span>
  </div>
  {{if eq .Tab "cnn1"}}
  <h1>CNN1 — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.CNN1Fixture.Version}}</strong>
    · {{.CNN1.ReportCount}} pipeline reports
    · {{.CNN1Loom.EntityFileCount}} <code>.entity</code>
    · {{.CNN1Loom.LoomReportCount}} loom reports
    · {{.CNN1Loom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>1D conv bedrock.</strong> {{.CNN1Fixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/cnn1</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "cnn2"}}
  <h1>CNN2 — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.CNN2Fixture.Version}}</strong>
    · {{.CNN2.ReportCount}} pipeline reports
    · {{.CNN2Loom.EntityFileCount}} <code>.entity</code>
    · {{.CNN2Loom.LoomReportCount}} loom reports
    · {{.CNN2Loom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>2D conv bedrock (NCHW).</strong> {{.CNN2Fixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/cnn2</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "cnn3"}}
  <h1>CNN3 — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.CNN3Fixture.Version}}</strong>
    · {{.CNN3.ReportCount}} pipeline reports
    · {{.CNN3Loom.EntityFileCount}} <code>.entity</code>
    · {{.CNN3Loom.LoomReportCount}} loom reports
    · {{.CNN3Loom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>3D conv bedrock (NCDHW).</strong> {{.CNN3Fixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/cnn3</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "lstm"}}
  <h1>LSTM — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.LSTMFixture.Version}}</strong>
    · {{.LSTM.ReportCount}} pipeline reports
    · {{.LSTMLoom.EntityFileCount}} <code>.entity</code>
    · {{.LSTMLoom.LoomReportCount}} loom reports
    · {{.LSTMLoom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>LSTM bedrock (Loom gate layout).</strong> {{.LSTMFixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/lstm</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "rnn"}}
  <h1>RNN — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.RNNFixture.Version}}</strong>
    · {{.RNN.ReportCount}} pipeline reports
    · {{.RNNLoom.EntityFileCount}} <code>.entity</code>
    · {{.RNNLoom.LoomReportCount}} loom reports
    · {{.RNNLoom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>RNN bedrock (tanh cell).</strong> {{.RNNFixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/rnn</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "layernorm"}}
  <h1>LayerNorm — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.LayerNormFixture.Version}}</strong>
    · {{.LayerNorm.ReportCount}} pipeline reports
    · {{.LayerNormLoom.EntityFileCount}} <code>.entity</code>
    · {{.LayerNormLoom.LoomReportCount}} loom reports
    · {{.LayerNormLoom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>LayerNorm bedrock (gamma + beta).</strong> {{.LayerNormFixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/layernorm</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "embedding"}}
  <h1>Embedding — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.EmbeddingFixture.Version}}</strong>
    · {{.Embedding.ReportCount}} pipeline reports
    · {{.EmbeddingLoom.EntityFileCount}} <code>.entity</code>
    · {{.EmbeddingLoom.LoomReportCount}} loom reports
    · {{.EmbeddingLoom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>Embedding bedrock (token lookup).</strong> {{.EmbeddingFixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/embedding</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "mixer"}}
  <h1>Mixer — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.MixerFixture.Version}}</strong>
    · {{.Mixer.ReportCount}} pipeline reports
    · {{.MixerLoom.EntityFileCount}} <code>.entity</code>
    · {{.MixerLoom.LoomReportCount}} loom reports
    · {{.MixerLoom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>All-layers mixer bedrock.</strong> {{.MixerFixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/mixer</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else if eq .Tab "mha"}}
  <h1>MHA — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.MHAFixture.Version}}</strong>
    · {{.MHA.ReportCount}} pipeline reports
    · {{.MHALoom.EntityFileCount}} <code>.entity</code>
    · {{.MHALoom.LoomReportCount}} loom reports
    · {{.MHALoom.PendingLoomSteps}} pending stream
  </div>
  <div class="fixture-banner"><strong>MHA bedrock (causal + RoPE).</strong> {{.MHAFixture.Note}}</div>
  <div class="pipeline-hint">
    Planets: <strong>pytorch · tensorflow · jax</strong>.
  Stream: <code>POST /api/v1/loom/stream/mha</code> · <a href="/PROGRESS.md">PROGRESS.md</a>
  </div>
  {{else}}
  <h1>Dense — planet pipeline compare</h1>
  <div class="meta">
    fixture <strong>{{.Fixture.Version}}</strong>
    · seed <strong>{{.Fixture.Seed}}</strong>
    · {{.Dense.ReportCount}} pipeline reports
    · {{.Loom.EntityFileCount}} <code>.entity</code> checkpoints
    · {{.Loom.LoomReportCount}} loom infer reports
    · {{.Loom.PendingLoomSteps}} planets pending stream
  </div>
  <div class="fixture-banner">
    <strong>Same training data per planet.</strong> {{.Fixture.Note}}
  </div>
  <div class="pipeline-hint">
    Per planet: <strong>native</strong> (train + infer) → <strong>export</strong> (reload saved format) and
    <strong>loom / entity</strong> (Python streams dense layers to <code>POST /api/v1/loom/stream</code>, Go builds <code>.entity</code>, Loom infer) —
    <em>parallel branches from native</em>, not onnx→safetensors→entity.
    Planets in scope: <strong>pytorch · tensorflow · jax · sklearn</strong> (paddle disabled).
    We do <em>not</em> cross-compare TensorFlow vs PyTorch — only steps within each planet's pipeline.
    See <a href="/PROGRESS.md">PROGRESS.md</a>.
    FP32 drift below {{fp32Tol}} shows as <strong>PASS</strong>.
  </div>
  {{if gt .Loom.EntityFileCount 0}}
  <div class="loom-stats">
    Layer stream is live. Re-run <code>./python/dense/run_engine.sh &lt;planet&gt;</code> with the host up to fill pending loom rows.
  </div>
  {{end}}
  {{end}}
</header>

<main>
{{if eq .Tab "cnn1"}}
{{if .CNN1LoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.CNN1ModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .CNN1LoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .CNN1.Models}}
  <div class="empty">No CNN1 reports yet.<br><br><code>go run .</code><br><code>./python/cnn1/run_cnn1.sh</code></div>
{{else}}
  {{range .CNN1.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "cnn2"}}
{{if .CNN2LoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.CNN2ModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .CNN2LoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .CNN2.Models}}
  <div class="empty">No CNN2 reports yet.<br><br><code>go run .</code><br><code>./python/cnn2/run_cnn2.sh</code></div>
{{else}}
  {{range .CNN2.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "cnn3"}}
{{if .CNN3LoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.CNN3ModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .CNN3LoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .CNN3.Models}}
  <div class="empty">No CNN3 reports yet.<br><br><code>go run .</code><br><code>./python/cnn3/run_cnn3.sh</code></div>
{{else}}
  {{range .CNN3.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "lstm"}}
{{if .LSTMLoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.LSTMModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .LSTMLoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .LSTM.Models}}
  <div class="empty">No LSTM reports yet.<br><br><code>go run .</code><br><code>./python/lstm/run_lstm.sh</code></div>
{{else}}
  {{range .LSTM.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "rnn"}}
{{if .RNNLoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.RNNModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .RNNLoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .RNN.Models}}
  <div class="empty">No RNN reports yet.<br><br><code>go run .</code><br><code>./python/rnn/run_rnn.sh</code></div>
{{else}}
  {{range .RNN.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "embedding"}}
{{if .EmbeddingLoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.EmbeddingModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .EmbeddingLoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .Embedding.Models}}
  <div class="empty">No Embedding reports yet.<br><br><code>go run .</code><br><code>./python/embedding/run_embedding.sh</code></div>
{{else}}
  {{range .Embedding.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "layernorm"}}
{{if .LayerNormLoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.LayerNormModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .LayerNormLoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .LayerNorm.Models}}
  <div class="empty">No LayerNorm reports yet.<br><br><code>go run .</code><br><code>./python/layernorm/run_layernorm.sh</code></div>
{{else}}
  {{range .LayerNorm.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "mixer"}}
{{if .MixerLoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.MixerModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .MixerLoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .Mixer.Models}}
  <div class="empty">No mixer reports yet.<br><br><code>go run .</code><br><code>./python/mixer/run_mixer.sh</code></div>
{{else}}
  {{range .Mixer.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else if eq .Tab "mha"}}
{{if .MHALoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.MHAModelsDir}})</h2>
  <table>
    <thead><tr><th>Model</th><th>Planet</th><th>.entity</th><th>Loom report</th></tr></thead>
    <tbody>
      {{range .MHALoomRows}}{{range .Planets}}
      <tr>
        <td>{{.ModelID}}</td><td>{{.Engine}}</td>
        <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
        <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
      </tr>
      {{end}}{{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .MHA.Models}}
  <div class="empty">No MHA reports yet.<br><br><code>go run .</code><br><code>./python/mha/run_mha.sh</code></div>
{{else}}
  {{range .MHA.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">{{range .Steps}}<span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>{{end}}</div>
      <table>
        <thead><tr><th>Compare</th><th>From</th><th>To</th><th>Max abs diff</th><th>Mean abs diff</th></tr></thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{else}}
{{if .LoomRows}}
<section class="loom-catalog">
  <h2>Loom entity checkpoints ({{.ModelsDir}})</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Planet</th>
        <th>.entity</th>
        <th>Loom report</th>
      </tr>
    </thead>
    <tbody>
      {{range .LoomRows}}
        {{range .Planets}}
        <tr>
          <td>{{.ModelID}}</td>
          <td>{{.Engine}}</td>
          <td class="mono">{{if .EntityFiles}}{{index .EntityFiles 0}}{{else}}—{{end}}</td>
          <td>{{if .HasLoomReport}}<span class="loom-yes">✓</span>{{else}}<span class="loom-no">pending</span>{{end}}</td>
        </tr>
        {{end}}
      {{end}}
    </tbody>
  </table>
</section>
{{end}}
{{if not .Dense.Models}}
  <div class="empty">
    No pipeline reports yet.<br><br>
    <code>go run .</code><br>
    <code>./python/dense/run_dense.sh</code>
  </div>
{{else}}
  {{range .Dense.Models}}
  <section class="model">
    <h2>{{.ModelID}}</h2>
    {{range .Pipelines}}
    <div class="planet-block">
      <div class="planet-head">{{.Planet}} pipeline</div>
      <div class="steps">
        {{range .Steps}}
        <span class="{{stepClass .Stage}}">{{stepLabel .Stage .Format}} ✓</span>
        {{if eq .Stage "loom"}}{{range .ArtifactPaths}}<span class="artifact">{{.}}</span>{{end}}{{end}}
        {{end}}
      </div>
      <table>
        <thead>
          <tr>
            <th>Compare</th>
            <th>From</th>
            <th>To</th>
            <th>Max abs diff</th>
            <th>Mean abs diff</th>
          </tr>
        </thead>
        <tbody>
          {{range .Compare}}
          <tr class="{{compareClass .}}">
            <td><span class="badge {{compareClass .}}">{{compareLabel .}}</span></td>
            <td>{{.FromStage}} / {{.FromFormat}}</td>
            <td>{{toLabel .}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MaxAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MaxAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MaxAbsDiff}}</div></div>{{end}}</td>
            <td>{{if .Pending}}—{{else}}<div class="diff-num"><div class="diff-sci">{{fmtSci .MeanAbsDiff}}</div><div class="diff-plain">≈ {{fmtDiffPlain .MeanAbsDiff}}</div><div class="diff-hint">{{fmtDiffHint .MeanAbsDiff}}</div></div>{{end}}</td>
          </tr>
          {{end}}
        </tbody>
      </table>
    </div>
    {{end}}
  </section>
  {{end}}
{{end}}
{{end}}
</main>

<footer>
  JSON: <a href="/api/v1/compare">/api/v1/compare</a>
  · export: <a href="/api/v1/export/all.pdf" download="planetbridging-compare-all.pdf">/api/v1/export/all.pdf</a>
  · loom catalog: <a href="/api/v1/loom/catalog">/api/v1/loom/catalog</a>
  · stream: <code>POST /api/v1/loom/stream</code>
  · progress: <a href="/PROGRESS.md">/PROGRESS.md</a>
</footer>
</body>
</html>`))
